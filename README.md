

This plugin provides functionality for generating few-shot prompts from Git repositories and GitHub issues. It can be used as a plugin for the LLM CLI tool.



1. Generate few-shot prompts from multiple Git repositories
2. Generate prompts from GitHub issues with optional chain-of-thought (COT) creation
3. Filter GitHub issue comments by issue owner or repository owner





```
llm engineer repo2fewshot <repo1_path> <repo2_path> ... [OPTIONS]
```

Options:
- `--output`, `-o`: Output file path for the generated few-shot prompt XML (default: llm-plugins/few_shot_prompt_all.xml)
- `--max-workers`, `-w`: Maximum number of worker threads for parallel processing (default: 5)
- `--exclude`, `-e`: Exclude patterns (e.g., '*.test.py', 'test_*.py')



```
llm engineer issue2fewshot <repo_url> <issue_number> <output_file> [OPTIONS]
```

Options:
- `--cot`: Include chain-of-thought in the prompt
- `--only-issue-owner`: Consider only comments from the issue owner
- `--only-repo-owner`: Consider only comments from the repo owner



Generate few-shot prompts from multiple repositories:
```
llm engineer repo2fewshot https://github.com/user/repo1 /path/to/local/repo2 -o output.xml -w 3 -e "*.test.py" -e "test_*.py"
```

Generate a prompt from a GitHub issue:
```
llm engineer issue2fewshot https://github.com/user/repo 123 issue_prompt.txt --cot --only-issue-owner
```



Contributions are welcome! Please feel free to submit a Pull Request.



This project is licensed under the MIT License.
