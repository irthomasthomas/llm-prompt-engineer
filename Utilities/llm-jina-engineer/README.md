# llm-jina

[![PyPI](https://img.shields.io/pypi/v/llm-jina.svg)](https://pypi.org/project/llm-jina/)
[![Changelog](https://img.shields.io/github/v/release/yourusername/llm-jina?include_prereleases&label=changelog)](https://github.com/yourusername/llm-jina/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/yourusername/llm-jina/blob/main/LICENSE)

LLM plugin for using Jina AI Search Foundation APIs

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-jina
```

## Usage

This plugin adds a new `jina-engineer` command to LLM. Before using the command, make sure to set your Jina AI API key as an environment variable:

```bash
export JINA_API_KEY=your_api_key_here
```

You can get your Jina AI API key for free at https://jina.ai/?sui=apikey

The `jina-engineer` command supports various operations:

1. Web scraping:
```bash
llm jina-engineer --url https://example.com "Extract main content"
```

2. Web search:
```bash
llm jina-engineer --search "Latest news about AI"
```

3. Text classification:
```bash
llm jina-engineer --classify --labels "positive,negative,neutral" "This product is amazing!"
```

4. Generate embeddings:
```bash
llm jina-engineer --embed "Embed this text"
```

5. Rerank search results:
```bash
llm jina-engineer --search --rerank "Best restaurants in New York"
```

6. Verify factual accuracy:
```bash
llm jina-engineer --ground "The Earth is flat"
```

7. Segment text:
```bash
llm jina-engineer --segment "This is a long text that needs to be segmented into smaller chunks."
```

You can combine multiple operations in a single command. For example:

```bash
llm jina-engineer --search --rerank --embed "AI advancements in 2023"
```

## Options

- `--url`: Specify a URL for web scraping
- `--search`: Perform a web search
- `--classify`: Classify text
- `--embed`: Generate embeddings
- `--rerank`: Rerank search results (requires --search)
- `--ground`: Verify factual accuracy
- `--segment`: Segment text
- `--model`: Specify the model to use for embeddings or classification (default: jina-embeddings-v3)
- `--labels`: Provide labels for classification (comma-separated)

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-jina
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
pip install -e '.[test]'
```

To run the tests:

```bash
pytest
```

## Contributing

Contributions to this plugin are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes and add tests
4. Run the tests to ensure everything is working
5. Submit a pull request

## License

This plugin is released under the Apache 2.0 License.