import click
import git
import xml.etree.ElementTree as ET
from pathlib import Path
import llm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from github import Github
import os
import re

FEW_SHOT_PROMPT_FILE = Path("few-shot-prompts/few_shot_prompt_all.xml")

def clone_or_update_repo(repo_url, base_dir):
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = base_dir / repo_name
    
    if repo_path.exists():
        repo = git.Repo(repo_path)
        origin = repo.remotes.origin
        origin.pull()
    else:
        git.Repo.clone_from(repo_url, repo_path)
    
    return repo_path

def process_repo(repo, base_dir):
    if repo.startswith("http") or repo.startswith("git@"):
        return clone_or_update_repo(repo, base_dir)
    else:
        repo_path = Path(repo)
        return repo_path if repo_path.is_dir() else None

def generate_few_shot_prompt(repo_paths):
    few_shot_prompt = ET.Element("few_shot_prompt")
    few_shot_prompt.text = "\n"
    
    for repo_path in repo_paths:
        if repo_path.is_dir():
            plugin = ET.SubElement(few_shot_prompt, "plugin")
            plugin.text = "\n"
            plugin.set("name", repo_path.name)
            
            readme_path = repo_path / "README.md"
            if readme_path.exists():
                readme = ET.SubElement(plugin, "readme")
                readme.text = "\n  " + readme_path.read_text().replace("\n", "\n  ") + "\n"
                readme.tail = "\n"
            
            for py_file in repo_path.rglob("*.py"):
                python_file = ET.SubElement(plugin, "python_file")
                python_file.set("name", str(py_file.relative_to(repo_path)))
                python_file.text = "\n  " + py_file.read_text().replace("\n", "\n  ") + "\n"
                python_file.tail = "\n"
            
            plugin.tail = "\n"
    
    return ET.tostring(few_shot_prompt, encoding="unicode", xml_declaration=True)

def fetch_github_issue(repo_owner, repo_name, issue_number):
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN environment variable is not set")

    g = Github(github_token)
    repo = g.get_repo(f"{repo_owner}/{repo_name}")
    issue = repo.get_issue(issue_number)
    return issue

def filter_comments(comments, only_issue_owner=False, only_repo_owner=False):
    if not (only_issue_owner or only_repo_owner):
        return comments

    filtered_comments = []
    issue_owner = comments[0].user.login
    repo_owner = comments[0].repository.owner.login

    for comment in comments:
        if only_issue_owner and comment.user.login == issue_owner:
            filtered_comments.append(comment)
        elif only_repo_owner and comment.user.login == repo_owner:
            filtered_comments.append(comment)

    return filtered_comments

def fetch_code_snippet(url):
    match = re.search(r'github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)#L(\d+)(?:-L(\d+))?', url)
    if not match:
        return None

    owner, repo, branch, path, start_line, end_line = match.groups()
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    response = requests.get(raw_url)
    if response.status_code != 200:
        return None

    content = response.text.splitlines()
    start_line = int(start_line)
    end_line = int(end_line) if end_line else start_line
    return '\n'.join(content[start_line-1:end_line])

def generate_cot_prompt(issue_title, issue_body):
    llm_model = llm.get_model()
    prompt = f"""Generate a chain-of-thought (CoT) prompt for the following issue:

Title: {issue_title}
Body: {issue_body}

The CoT prompt should guide the model through the process of understanding and solving the problem presented in the issue. Include steps for analysis, problem-solving, and implementation considerations.

CoT Prompt:"""

    response = llm_model.prompt(prompt)
    return response.text()

def generate_issue_based_prompt(issue, comments, include_cot=False, only_issue_owner=False, only_repo_owner=False):
    filtered_comments = filter_comments(comments, only_issue_owner, only_repo_owner)
    
    few_shot_prompt = ET.Element("few_shot_prompt")
    few_shot_prompt.text = "\n"
    
    issue_element = ET.SubElement(few_shot_prompt, "issue")
    issue_element.set("title", issue.title)
    issue_element.text = "\n  " + issue.body.replace("\n", "\n  ") + "\n"
    
    if include_cot:
        cot_prompt = generate_cot_prompt(issue.title, issue.body)
        cot_element = ET.SubElement(few_shot_prompt, "chain_of_thought")
        cot_element.text = "\n  " + cot_prompt.replace("\n", "\n  ") + "\n"
    
    for comment in filtered_comments:
        comment_element = ET.SubElement(few_shot_prompt, "comment")
        comment_element.set("author", comment.user.login)
        comment_text = comment.body
        
        # Check for code snippet links and fetch them
        for line in comment_text.splitlines():
            if "github.com" in line and "/blob/" in line:
                snippet = fetch_code_snippet(line)
                if snippet:
                    comment_text = comment_text.replace(line, f"{line}\n\nCode snippet:\n{snippet}")
        
        comment_element.text = "\n  " + comment_text.replace("\n", "\n  ") + "\n"
    
    return ET.tostring(few_shot_prompt, encoding="unicode", xml_declaration=True)

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("repos", nargs=-1)
    @click.option("--output", "-o", default=str(FEW_SHOT_PROMPT_FILE), help="Output file path for the generated few-shot prompt XML")
    @click.option("--max-workers", "-w", default=5, help="Maximum number of worker threads for parallel processing")
    def repo2fewshot(repos, output, max_workers):
        """Generate a few-shot prompt file from a list of repository URLs or local paths for llm plugins."""
        base_dir = Path.home() / ".cache" / "llm-fewshot-generator" / "repos"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        repo_paths = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_repo = {executor.submit(process_repo, repo, base_dir): repo for repo in repos}
            for future in as_completed(future_to_repo):
                repo = future_to_repo[future]
                try:
                    repo_path = future.result()
                    if repo_path:
                        repo_paths.append(repo_path)
                    else:
                        click.echo(f"Warning: {repo} is not a valid repository or directory. Skipping.")
                except Exception as exc:
                    click.echo(f"Error processing {repo}: {exc}")
        
        if not repo_paths:
            click.echo("Error: No valid repositories or directories provided.")
            return
        
        few_shot_prompt = generate_few_shot_prompt(repo_paths)
        
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            f.write(few_shot_prompt)
        
        click.echo(f"Few-shot prompt XML generated and saved to {output_path}")

    @cli.command()
    @click.argument("repo_owner")
    @click.argument("repo_name")
    @click.argument("issue_number", type=int)
    @click.option("--output", "-o", default="issue_prompt.xml", help="Output file path for the generated issue-based prompt XML")
    @click.option("--cot", is_flag=True, help="Include chain-of-thought prompt")
    @click.option("--only-issue-owner", is_flag=True, help="Only include comments from the issue owner")
    @click.option("--only-repo-owner", is_flag=True, help="Only include comments from the repo owner")
    def issue2fewshot(repo_owner, repo_name, issue_number, output, cot, only_issue_owner, only_repo_owner):
        """Generate a few-shot prompt file from a GitHub issue."""
        try:
            issue = fetch_github_issue(repo_owner, repo_name, issue_number)
            comments = list(issue.get_comments())
            
            few_shot_prompt = generate_issue_based_prompt(
                issue, 
                comments, 
                include_cot=cot, 
                only_issue_owner=only_issue_owner, 
                only_repo_owner=only_repo_owner
            )
            
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with output_path.open("w", encoding="utf-8") as f:
                f.write(few_shot_prompt)
            
            click.echo(f"Issue-based prompt XML generated and saved to {output_path}")
        except Exception as e:
            click.echo(f"Error: {str(e)}")

if __name__ == "__main__":
    register_commands(click.Group())