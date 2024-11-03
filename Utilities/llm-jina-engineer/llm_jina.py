import click
import llm
import os
import requests
import json
from typing import Dict, List, Union

# Get your Jina AI API key for free: https://jina.ai/?sui=apikey

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("query", nargs=-1)
    @click.option("--url", help="URL for web scraping")
    @click.option("--search", is_flag=True, help="Perform a web search")
    @click.option("--classify", is_flag=True, help="Classify text or image")
    @click.option("--embed", is_flag=True, help="Generate embeddings")
    @click.option("--rerank", is_flag=True, help="Rerank search results")
    @click.option("--ground", is_flag=True, help="Verify factual accuracy")
    @click.option("--segment", is_flag=True, help="Segment text")
    @click.option("--model", default="jina-embeddings-v3", help="Model to use for embeddings or classification")
    @click.option("--labels", help="Labels for classification, comma-separated")
    def jina_engineer(query, url, search, classify, embed, rerank, ground, segment, model, labels):
        """Use Jina AI Search Foundation APIs for various tasks."""
        api_key = os.environ.get("JINA_API_KEY")
        if not api_key:
            raise click.ClickException("Please set the JINA_API_KEY environment variable.")

        query_text = " ".join(query)

        try:
            if url:
                content = read_url(api_key, url)
                click.echo(f"Content from {url}:\n{content}")

            if search:
                results = perform_search(api_key, query_text)
                click.echo(f"Search results:\n{json.dumps(results, indent=2)}")

            if classify:
                if not labels:
                    raise click.ClickException("Please provide labels for classification using --labels")
                label_list = [label.strip() for label in labels.split(",")]
                classification = classify_text(api_key, query_text, label_list, model)
                click.echo(f"Classification results:\n{json.dumps(classification, indent=2)}")

            if embed:
                embeddings = generate_embeddings(api_key, query_text, model)
                click.echo(f"Embeddings:\n{json.dumps(embeddings, indent=2)}")

            if rerank:
                if not search:
                    raise click.ClickException("Reranking requires search results. Please use --search along with --rerank")
                reranked = rerank_results(api_key, query_text, results)
                click.echo(f"Reranked results:\n{json.dumps(reranked, indent=2)}")

            if ground:
                grounding = verify_statement(api_key, query_text)
                click.echo(f"Grounding results:\n{json.dumps(grounding, indent=2)}")

            if segment:
                segments = segment_text(api_key, query_text)
                click.echo(f"Segmentation results:\n{json.dumps(segments, indent=2)}")

        except requests.RequestException as e:
            raise click.ClickException(f"API request failed: {str(e)}")

def read_url(api_key: str, url: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    response = requests.post("https://r.jina.ai/", headers=headers, json={"url": url})
    response.raise_for_status()
    return response.json()["data"]["content"]

def perform_search(api_key: str, query: str) -> Dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    response = requests.post("https://s.jina.ai/", headers=headers, json={"q": query})
    response.raise_for_status()
    return response.json()["data"]

def classify_text(api_key: str, text: str, labels: List[str], model: str) -> Dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "model": model,
        "input": [text],
        "labels": labels
    }
    response = requests.post("https://api.jina.ai/v1/classify", headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def generate_embeddings(api_key: str, text: str, model: str) -> Dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "model": model,
        "input": [text]
    }
    response = requests.post("https://api.jina.ai/v1/embeddings", headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def rerank_results(api_key: str, query: str, documents: List[Dict]) -> Dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "documents": [doc["content"] for doc in documents]
    }
    response = requests.post("https://api.jina.ai/v1/rerank", headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def verify_statement(api_key: str, statement: str) -> Dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    response = requests.post("https://g.jina.ai/", headers=headers, json={"statement": statement})
    response.raise_for_status()
    return response.json()["data"]

def segment_text(api_key: str, text: str) -> Dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "content": text,
        "return_chunks": True
    }
    response = requests.post("https://segment.jina.ai/", headers=headers, json=data)
    response.raise_for_status()
    return response.json()