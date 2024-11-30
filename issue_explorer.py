#!/usr/bin/env python3
import json
import sys
import random
from typing import List, Dict
from collections import defaultdict

class IssueExplorer:
    def __init__(self, json_file='simonw_programming_examples.json'):
        """Initialize the Issue Explorer with collected data."""
        try:
            with open(json_file, 'r') as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data['examples'])} issues from {json_file}")
        except FileNotFoundError:
            print(f"Error: File {json_file} not found.")
            sys.exit(1)

    def analyze_and_categorize(self):
        """Analyze issues and suggest categories based on content."""
        categories = defaultdict(list)
        for issue in self.data['examples']:
            category = issue.get('category', 'Uncategorized')
            categories[category].append(issue)
        return dict(categories)

    def display_issue_details(self, issue: Dict):
        """Display detailed information about an issue."""
        print(f"\nTitle: {issue.get('title', 'Unknown Title')}")
        print(f"Repository: {issue.get('repository', 'Unknown Repository')}")
        print(f"Issue Number: {issue.get('issue_number', 'Unknown Number')}")
        print(f"Category: {issue.get('category', 'Uncategorized')}")
        print(f"URL: {issue.get('url', 'Unknown URL')}")
        print(f"Gemini Rating: {issue.get('gemini_rating', 'N/A')}/10")
        print("\nGemini's Analysis:")
        print(issue.get('gemini_explanation', 'No explanation available').strip())
        print("-" * 80)

    def list_categories(self) -> Dict[str, List[Dict]]:
        """List all categories and their issues."""
        categories = self.analyze_and_categorize()
        print("\nCategories and Issue Counts:")
        for category, issues in categories.items():
            print(f"{category}: {len(issues)} issues")
        return categories

    def view_category(self, category: str):
        """View all issues in a specific category."""
        categories = self.analyze_and_categorize()
        if category in categories:
            print(f"\nIssues in category '{category}':")
            for issue in categories[category]:
                self.display_issue_details(issue)
        else:
            print(f"No issues found in category '{category}'")

    def search_issues(self, keyword: str):
        """Search issues by keyword."""
        found_issues = []
        for issue in self.data['examples']:
            if (keyword.lower() in issue.get('title', '').lower() or
                keyword.lower() in issue.get('gemini_explanation', '').lower()):
                found_issues.append(issue)
        if found_issues:
            print(f"\nFound {len(found_issues)} issues matching '{keyword}':")
            for issue in found_issues:
                self.display_issue_details(issue)
        else:
            print(f"No issues found matching '{keyword}'")

    def generate_training_set(self, count: int = 5):
        """Generate a random selection of issues for training."""
        data_count = len(self.data['examples'])
        if count > data_count:
            count = data_count
        selected_issues = random.sample(self.data['examples'], count)
        print(f"\nRandom selection of {count} issues for training:")
        for issue in selected_issues:
            self.display_issue_details(issue)

def main():
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = 'simonw_programming_examples.json'
    explorer = IssueExplorer(json_file)
    
    while True:
        print("\nGitHub Issue Explorer")
        print("1. List Categories")
        print("2. View Issues by Category")
        print("3. Search Issues")
        print("4. Generate Training Set")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            explorer.list_categories()
        
        elif choice == '2':
            explorer.list_categories()
            category = input("\nEnter category name: ")
            explorer.view_category(category)
        
        elif choice == '3':
            keyword = input("\nEnter search keyword: ")
            explorer.search_issues(keyword)
        
        elif choice == '4':
            try:
                count = int(input("\nHow many issues in training set? "))
                explorer.generate_training_set(count)
            except ValueError:
                print("Please enter a valid number")
        
        elif choice == '5':
            print("\nThank you for using GitHub Issue Explorer!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()