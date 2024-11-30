#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
import requests
from github import Github, Event, GithubException

# Todo: issues are already retrieved in descending order of date - stop checking them as soon as a date is older than the cut-off.

class GithubIssueAnalyzer:
    def __init__(self, username, github_token=None, min_quality_score=6, max_dataset_size=5, languages=None):
        """
        Initialize the GitHub Issue Analyzer.
        
        :param username: GitHub username to analyze
        :param github_token: GitHub personal access token (optional)
        :param min_quality_score: Minimum quality score for issues to be included in the dataset
        :param max_dataset_size: Maximum number of issues to include in the dataset
        """
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        if not self.github_token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN environment variable.")
        self.g = Github(self.github_token)
        self.username = username
        self.user = self.g.get_user(username)
        self.min_quality_score = min_quality_score
        self.max_dataset_size = max_dataset_size
        self.languages = languages or ['Python']  # Make language filtering configurable


    def filter_repositories(self):
        """
        Filter repositories using GitHub's search API.
        
        :return: List of qualifying repositories
        """
        qualifying_repos = []
        
        # Construct search query
        query = f'user:{self.username} fork:false is:public archived:false'
        
        # Add language filter if specified
        if self.languages:
            query += f' language:{" language:".join(self.languages)}'
            
        try:
            # Use search API to filter efficiently
            repos = self.g.search_repositories(
                query=query,
                sort='updated',
                order='desc'
            )
            
            # Apply remaining filters on pre-filtered results
            for repo in repos:
                if (repo.stargazers_count >= 20 and 
                    repo.size > 100 and 
                    repo.has_issues and
                    repo.updated_at > datetime.now() - timedelta(days=365)):
                    
                    qualifying_repos.append(repo)
                    print(f"Repository {repo.name} qualifies")
                    
                    # Stop once we have enough repos
                    if len(qualifying_repos) >= self.max_dataset_size:
                        break
                        
            print(f"Total qualifying repositories: {len(qualifying_repos)}")
            return qualifying_repos
            
        except Exception as e:
            print(f"Error searching repositories: {e}")
            return []
        

    def find_qualifying_issues(self):
        """
        Find and filter GitHub issues based on specified criteria.
        
        :return: List of qualifying issues
        """
        qualifying_issues = []
        two_years_ago = datetime.now() - timedelta(days=365)

        repos = self.filter_repositories()
        total_issues_checked = 0
        total_qualifying_issues = 0

        print("Finding qualifying issues...")
        for repo in repos:
            try:
                issues = repo.get_issues(state='closed', sort='closed', direction='desc')
                print(f"Checking issues in repository {repo.name}")

                for issue in issues:
                    total_issues_checked += 1
                    print(f"Checking issue {total_issues_checked}: {issue.title}")

                    if issue.closed_at < two_years_ago:
                        print(f"Issue {issue.title} is older than 2 years, skipping")
                        continue

                    if issue.user.login == self.username:
                        if issue.comments >= 4:
                            if self._is_interesting_issue(issue):
                                qualifying_issues.append(issue)
                                total_qualifying_issues += 1
                                print(f"Issue {issue.title} qualifies")
            except Exception as e:
                print(f"Error processing repo {repo.name}: {str(e)}")

        print(f"Total qualifying issues: {total_qualifying_issues}")
        return qualifying_issues

    def _is_interesting_issue(self, issue):
        """
        Check if an issue involves feature engineering, refactoring, or problem-solving.
        
        :param issue: GitHub issue object
        :return: Boolean indicating issue's interestingness
        """
        interesting_keywords = [
            'enhancement', 'strategy', 'approach', 'solution',
            'architecture', 'performance', 'feature', 'solve',
            'refactor', 'improvement', 'optimization', 'design',
            'pattern', 'prompt', 'support', 'implement', 'fix'             
        ]

        text = f"{issue.title} {issue.body}".lower()
        is_interesting = any(keyword in text for keyword in interesting_keywords)
        return is_interesting

    def _load_rubric_prompt(self):
        """
        Load the rubric prompt from XML file.
        
        :return: Rubric prompt string
        """
        prompt_path = os.path.join(os.path.dirname(__file__), "rubric_prompt.xml")
        print(f"Loading rubric prompt from {prompt_path}")

        with open(prompt_path, 'r') as f:
            return f.read()

    def analyze_issue_with_gemini(self, issue):
        """
        Analyze an issue using the Gemini model via LLM command.
        
        :param issue: GitHub issue object
        :return: Analysis result dictionary
        """
        try:
            # Use the new method to construct the issue thread with sorted events
            issue_thread = self.construct_issue_thread_with_sorted_events(issue)
            
            prompt_template = """Here is the GitHub issue thread and its context that you need to analyze:

<issue_thread>
{{ISSUE_THREAD}}
</issue_thread>
"""
            system_prompt = self._load_rubric_prompt()
            prompt = prompt_template.replace("{{ISSUE_THREAD}}", issue_thread)
            env = os.environ.copy()
            env['LLM_LOAD_PLUGINS'] = 'llm-gemini'
            print(f"Running Gemini analysis for issue {issue.title}")

            result = subprocess.run(
                ['llm', '-m', 'gemini-1.5-flash-latest', prompt, '--system', system_prompt],
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"LLM command failed with code {result.returncode}")
            # print trimmed result 
            print(result.stdout[:500])
            output = result.stdout.strip()
            analysis = self._extract_tag_content(output, 'analysis')
            justification = self._extract_tag_content(output, 'justification')
            rating = int(self._extract_tag_content(output, 'rating'))

            print(f"Analysis for issue {issue.title}: {analysis}")
            print(f"Justification for issue {issue.title}: {justification}")
            print(f"Rating for issue {issue.title}: {rating}")

            return {
                'rating': rating,
                'analysis': analysis,
                'justification': justification,
                'raw_output': output
            }
        except Exception as e:
            print(f"Analysis failed for issue {issue.title}: {str(e)}")
            return {
                'rating': 0,
                'analysis': 'Analysis failed',
                'justification': str(e),
                'raw_output': ''
            }
    
    
    def fetch_and_sort_events(self, issue):
        events = list(issue.get_events())
        comments = list(issue.get_comments())
        code_changes = []
    
        if issue.pull_request and issue.pull_request.get('url'):
            try:
                pr = self.g.get_repo(issue.repository.full_name).get_pull(issue.number)
                for commit in pr.get_commits():
                    code_changes.append({
                        'type': 'commit',
                        'created_at': commit.commit.author.date,
                        'sha': commit.sha,
                        'message': commit.commit.message,
                        'diff': self._get_commit_diff(commit)
                    })
                for comment in pr.get_review_comments():
                    code_changes.append({
                        'type': 'review_comment',
                        'created_at': comment.created_at,
                        'path': comment.path,
                        'position': comment.position,
                        'body': comment.body,
                        'diff_hunk': self._format_review_diff(comment.diff_hunk)
                    })
            except GithubException as e:
                print(f"Error fetching PR for issue {issue.number}: {e}")
        return sorted(events + comments + code_changes, key=lambda e: e.created_at)    

    
    def _get_commit_diff(self, commit):
        """
        Format commit diff into before/after blocks for each file.
        
        :param commit: GitHub commit object
        :return: List of dictionaries with file diffs
        """
        files = commit.files
        formatted_diffs = []
        for file in files:
            if file.patch:
                before, after = parse_diff(file.patch)
                formatted_diffs.append({
                    'file': file.filename,
                    'before': before,
                    'after': after
                })
        return formatted_diffs
    
    
    def _format_review_diff(self, diff_hunk):
        """
        Format review comment diff hunk into before and after code blocks.
        
        :param diff_hunk: Diff hunk string from review comment
        :return: Dictionary with before and after code blocks
        """
        before, after = parse_diff(diff_hunk)
        return {
            'before': before,
            'after': after
        }
        
    
    def _construct_issue_thread(self, issue):
        """
        Constructs a formatted issue thread text, including only comments by 'simonw'.
        
        :param issue: GitHub issue object
        :return: Formatted issue thread string
        """
        thread = f"Title: {issue.title}\n\nInitial Post by {issue.user.login}:\n{self._replace_code_urls_with_snippets(issue.body)}\n\n"
        for comment in issue.get_comments():
            if comment.user.login == self.username:
                comment_body = self._replace_code_urls_with_snippets(comment.body)
                comment_body = self._convert_diff_fences(comment_body)
                thread += f"Comment by {comment.user.login}:\n{comment_body}\n\n"
        thread = self._convert_diff_fences(thread)
        return thread

    
    def construct_issue_thread_with_sorted_events(self, issue):
        """
        Constructs a formatted issue thread text including sorted events.
        
        :param issue: GitHub issue object
        :return: Formatted issue thread string
        """
        sorted_events = self.fetch_and_sort_events(issue)
        thread = f"Title: {issue.title}\n\n"
        thread += f"Initial Post by {issue.user.login}:\n{self._replace_code_urls_with_snippets(issue.body)}\n\n"
        
        for event in sorted_events:
            if isinstance(event, dict):  # Code change event
                if event['type'] == 'commit':
                    thread += f"Commit {event['sha'][:7]} at {event['created_at']}:\n"
                    thread += f"{event['message']}\n\n"
                    for file_diff in event['diff']:
                        thread += f"File: {file_diff['file']}\n"
                        thread += f"Before:\n```\n{file_diff['before']}\n```\n"
                        thread += f"After:\n```\n{file_diff['after']}\n```\n\n"
                elif event['type'] == 'review_comment':
                    thread += f"Review comment at {event['created_at']} on {event['path']}:\n"
                    thread += f"{event['body']}\n"
                    diff = event['diff_hunk']
                    thread += f"Before:\n```\n{diff['before']}\n```\n"
                    thread += f"After:\n```\n{diff['after']}\n```\n\n"
            else:
                # Handle regular comments and other events as before
                if hasattr(event, 'body'):
                    body = self._replace_code_urls_with_snippets(event.body)
                    body = self._convert_diff_fences(body)
                    thread += f"Comment by {event.user.login} at {event.created_at}:\n{body}\n\n"
                else:
                    thread += f"Event: {event.event} at {event.created_at}\n\n"
                    
        return thread

    
    def _replace_code_urls_with_snippets(self, text):
        """
        Replace GitHub file URLs with actual code snippets.

        :param text: Text containing GitHub file URLs
        :return: Text with URLs replaced by code snippets
        """
        github_file_url_pattern = r"https://github.com/\S+/blob/\S+/(.+?)(#L\d+-L\d+)?"
        match = re.search(github_file_url_pattern, text)

        # Mapping of file extensions to language identifiers
        extension_to_language = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'sh': 'bash',
            'sql': 'sql',
            # Add more mappings as needed
        }

        while match:
            file_path = match.group(1)
            line_range = match.group(2)
            repo_name = re.search(r"https://github.com/([^/]+)/", text).group(1)
            repo = self.g.get_repo(repo_name)
            print(f"Fetching file {file_path} from repository {repo_name}")

            file_content = repo.get_contents(file_path)
            file_lines = file_content.decoded_content.decode("utf-8").splitlines()

            if line_range:
                start, end = map(int, line_range.strip("#L").split("-"))
                code_snippet = "\n".join(file_lines[start - 1 : end])
            else:
                code_snippet = "\n".join(file_lines)

            # Extract file extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lstrip('.').lower()
            language = extension_to_language.get(ext, '')

            if language:
                code_block = f"```{language}\n{code_snippet}\n```"
            else:
                code_block = f"```\n{code_snippet}\n```"

            text = text.replace(match.group(0), code_block)
            match = re.search(github_file_url_pattern, text)

        return text


    def _extract_tag_content(self, text, tag):
        """
        Extract content between XML-style tags.
        
        :param text: Input text containing XML-style tags
        :param tag: Tag to extract content from
        :return: Extracted content string
        """
        print(f"Extracting content for tag {tag}")
        print("Text:", text)
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""


    def categorize_issue(self, issue):
        """
        Categorize an issue based on its content.
        
        :param issue: GitHub issue object
        :return: Categorization string
        """
        categories = {
            'API Design': ['api', 'interface', 'endpoint'],
            'Performance Optimization': ['performance', 'speed', 'optimization', 'efficient'],
            'Bug Fixing': ['bug', 'fix', 'error', 'resolve'],
            'Feature Engineering': ['feature', 'enhancement', 'new', 'improve'],
            'Refactoring': ['refactor', 'restructure', 'clean', 'improve code']
        }

        text = f"{issue.title} {issue.body}".lower()

        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category

        return 'Miscellaneous'


    def extract_code_snippets(self, issue):
        """
        Extract code snippets from an issue.
        
        :param issue: GitHub issue object
        :return: List of code snippets
        """
        code_snippets = []
        code_pattern = r'```[\w]*\n(.*?)\n```'

        code_snippets.extend(re.findall(code_pattern, issue.body, re.DOTALL))

        for comment in issue.get_comments():
            code_snippets.extend(re.findall(code_pattern, comment.body, re.DOTALL))

        return code_snippets


    def generate_dataset(self):
        """
        Generate the complete dataset of programming examples.
        
        :return: JSON-compatible dataset
        """
        qualifying_issues = self.find_qualifying_issues()
        analyzed_issues = []

        print("Generating dataset...")
        for issue in qualifying_issues:
            gemini_analysis = self.analyze_issue_with_gemini(issue)

            if gemini_analysis['rating'] >= self.min_quality_score:
                example = {
                    'category': self.categorize_issue(issue),
                    'issue_number': issue.number,
                    'title': issue.title,
                    'repository': issue.repository.name,
                    'url': issue.html_url,
                    'gemini_explanation': gemini_analysis['analysis'],
                    'gemini_justification': gemini_analysis['justification'],
                    'gemini_rating': gemini_analysis['rating']
                }
                analyzed_issues.append(example)
                print(f"Included issue {issue.title} in dataset")

        analyzed_issues.sort(key=lambda x: x['gemini_rating'], reverse=True)
        analyzed_issues = analyzed_issues[:self.max_dataset_size]

        return {'examples': analyzed_issues}

    def _generate_xml_prompt(self, issue):
        """
        Generate an XML-like few-shot prompt for an issue.
        
        :param issue: GitHub issue object
        :return: XML-formatted string
        """
        issue_xml = f"""<example category="{self.categorize_issue(issue)}">
    <issue_title>{issue.title}</issue_title>
    <issue_body>{issue.body}</issue_body>
    <comments>{' '.join([comment.body for comment in issue.get_comments()])}</comments>
    <code>{'\\n'.join(self.extract_code_snippets(issue))}</code>
</example>"""
        return issue_xml

    def save_dataset(self, filename=None):
        """
        Save the generated dataset to a JSON file.
        
        :param filename: Output filename
        """
        if filename is None:
            filename = f"gemini_flash_programming_examples.json"
        dataset = self.generate_dataset()
        print(f"Saving dataset to {filename}")
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
    
    def _convert_diff_fences(self, text):
        diff_pattern = re.compile(r'```diff\s*(.*?)\s*```', re.DOTALL)
        def replace_diff_fence(match):
            diff_content = match.group(1)
            converted = convert_diff_to_code_blocks(diff_content)
            return converted
        return diff_pattern.sub(replace_diff_fence, text)


def parse_diff(diff_str):
    """
    Parse a unified diff string into before and after code blocks.
    
    :param diff_str: Unified diff string
    :return: Tuple containing before and after code blocks as strings
    :raises ValueError: If diff string is malformed
    """
    if not diff_str:
        return '', ''
        
    before_lines = []
    after_lines = []
    
    try:
        for line in diff_str.split('\n'):
            # Skip special diff headers
            if line.startswith('+++') or line.startswith('---') or line.startswith('@@'):
                continue
                
            # Handle different line types
            if line.startswith('-'):
                before_lines.append(line[1:])
            elif line.startswith('+'):
                after_lines.append(line[1:])
            elif line.startswith(' '):
                before_lines.append(line[1:])
                after_lines.append(line[1:])
            elif line.startswith('Binary files'):
                return '[Binary file]', '[Binary file]'
            
    except Exception as e:
        raise ValueError(f"Failed to parse diff: {str(e)}")
        
    return '\n'.join(before_lines), '\n'.join(after_lines)


def convert_diff_to_code_blocks(diff_str):
    """
    Convert a diff string into a series of formatted code block pairs.
    
    :param diff_str: Raw diff string
    :return: Formatted string with before/after code blocks
    """
    lines = diff_str.strip().split('\n')
    result = []
    current_block = None
    
    for line in lines:
        if line.startswith('diff'):
            # Complete previous block if exists
            if current_block and current_block.get('before') or current_block.get('after'):
                before_code = '\n'.join(current_block['before'])
                after_code = '\n'.join(current_block['after'])
                result.append(f"File: {current_block['file']}\n")
                result.append(f"Before:\n```\n{before_code}\n```\n")
                result.append(f"After:\n```\n{after_code}\n```\n")
            
            # Start new block
            file_name = line.split()[-1].split('/')[-1]
            current_block = {
                'file': file_name,
                'before': [],
                'after': []
            }
            
        elif current_block:
            # Skip headers
            if line.startswith('+++') or line.startswith('---') or line.startswith('@@'):
                continue
                
            # Process diff content
            if line.startswith('-'):
                current_block['before'].append(line[1:])
            elif line.startswith('+'):
                current_block['after'].append(line[1:])
            elif line.startswith(' '):
                current_block['before'].append(line[1:])
                current_block['after'].append(line[1:])
    
    # Handle last block
    if current_block and (current_block.get('before') or current_block.get('after')):
        before_code = '\n'.join(current_block['before'])
        after_code = '\n'.join(current_block['after'])
        result.append(f"File: {current_block['file']}\n")
        result.append(f"Before:\n```\n{before_code}\n```\n")
        result.append(f"After:\n```\n{after_code}\n```\n")
    
    return '\n'.join(result)


def main():
    """
    Main function to run the GitHub Issue Analyzer.
    """
    username = sys.argv[1] if len(sys.argv) > 1 else input("Enter GitHub username: ")
    analyzer = GithubIssueAnalyzer(username)
    analyzer.save_dataset()

if __name__ == '__main__':
    main()
