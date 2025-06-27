# pylint: disable=too-many-lines,too-many-branches,too-many-statements,too-many-locals,too-few-public-methods
"""
Script to create JIRA tickets from pull requests using AI assistance.
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional

import requests


class PullRequestFetcher:
    """Handles fetching pull request data from GitHub."""

    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()
        if self.github_token:
            self.session.headers.update(
                {
                    "Authorization": f"token {self.github_token}",
                    "Accept": "application/vnd.github.v3+json",
                }
            )

    def fetch_pr_details(self, pr_url: str) -> Dict[str, Any]:
        """Fetch pull request details from GitHub API."""
        if "github.com" not in pr_url or "/pull/" not in pr_url:
            raise ValueError(
                "Invalid GitHub PR URL format. Expected: https://github.com/owner/repo/pull/123"
            )

        parts = pr_url.rstrip("/").split("/")
        try:
            pr_number = int(parts[-1])
            owner = parts[-4]
            repo = parts[-3]
        except (IndexError, ValueError) as e:
            raise ValueError(
                "Could not parse PR URL. Expected format: https://github.com/owner/repo/pull/123"
            ) from e

        pr_api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        pr_response = self.session.get(pr_api_url)
        pr_response.raise_for_status()
        pr_data = pr_response.json()

        comments_url = (
            f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
        )
        comments_response = self.session.get(comments_url)
        comments_response.raise_for_status()
        comments_data = comments_response.json()

        review_comments_url = (
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        )
        review_comments_response = self.session.get(review_comments_url)
        review_comments_response.raise_for_status()
        review_comments_data = review_comments_response.json()

        return {
            "pr_data": pr_data,
            "comments": comments_data,
            "review_comments": review_comments_data,
        }


class AIClient:
    """Handles communication with AI endpoints."""

    PROVIDER_ENDPOINTS = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
    }

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        ai_model: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("AI_PROVIDER", "openai")).lower()

        if endpoint_url:
            self.endpoint_url = endpoint_url.rstrip("/")
        else:
            self.endpoint_url = self.PROVIDER_ENDPOINTS.get(
                self.provider, self.PROVIDER_ENDPOINTS["openai"]
            )

        if api_key:
            self.api_key = api_key
        elif self.provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("AI_TOKEN")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AI_TOKEN")

        if not self.api_key:
            raise ValueError(f"API key for {self.provider} not found.")

        default_models = {
            "openai": "gpt-4",
            "gemini": "gemini-1.5-flash",
        }
        self.ai_model = (
            ai_model or os.getenv("AI_MODEL") or default_models.get(self.provider)
        )

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def generate_jira_ticket(self, user_query: str, pr_context: str) -> str:
        """Generate JIRA ticket content using the specified AI provider."""
        system_prompt = """You are an expert at creating JIRA tickets from pull request information. 
Given a user query and pull request context, generate a well-structured JIRA ticket with:
- A clear, concise title
- A detailed description
- Appropriate priority and issue type suggestions
- Any relevant labels or components

Format the response as a structured JIRA ticket."""

        user_prompt = f"""
User Query: {user_query}

Pull Request Context:
{pr_context}

Please create a JIRA ticket based on this information.
"""

        if self.provider == "gemini":
            url = f"{self.endpoint_url}/{self.ai_model}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": f"{system_prompt}\n{user_prompt}"}]}]
            }
        else:  # OpenAI and compatible APIs
            url = self.endpoint_url
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
            payload = {
                "model": self.ai_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 1500,
                "temperature": 0.7,
            }

        response = self.session.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

        try:
            if self.provider == "gemini":
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise IOError(f"Unexpected API response format: {result}") from e


def format_pr_context(pr_details: Dict[str, Any]) -> str:
    """Format pull request details into a context string."""
    pr_data = pr_details["pr_data"]
    comments = pr_details["comments"]
    review_comments = pr_details["review_comments"]

    context = f"""
Pull Request #{pr_data["number"]}: {pr_data["title"]}
URL: {pr_data["html_url"]}
Author: {pr_data["user"]["login"]}
State: {pr_data["state"]}
Created: {pr_data["created_at"]}
Updated: {pr_data["updated_at"]}

Description:
{pr_data["body"] or "No description provided"}

Files Changed: {pr_data.get("changed_files", "N/A")}
Additions: +{pr_data.get("additions", "N/A")}
Deletions: -{pr_data.get("deletions", "N/A")}
"""

    if comments:
        context += "\n\nComments:\n"
        for comment in comments:
            context += f"- {comment['user']['login']}: {comment['body']}\n"

    if review_comments:
        context += "\n\nReview Comments:\n"
        for comment in review_comments:
            context += f"- {comment['user']['login']}: {comment['body']}\n"

    return context


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Create JIRA tickets from pull requests using AI"
    )
    parser.add_argument(
        "pr_url", help="Pull request URL (e.g., https://github.com/owner/repo/pull/123)"
    )
    parser.add_argument(
        "--ai-provider",
        choices=["openai", "gemini"],
        help="AI provider (can also be set via AI_PROVIDER env var)",
    )
    parser.add_argument(
        "--ai-token",
        help="AI API token (can also be set via AI_TOKEN, OPENAI_API_KEY, or GEMINI_API_KEY env vars)",
    )
    parser.add_argument(
        "--ai-endpoint",
        help="Custom AI API endpoint (overrides provider default, can be set via AI_ENDPOINT env var)",
    )
    parser.add_argument(
        "--github-token", help="GitHub API token (or set GITHUB_TOKEN env var)"
    )
    parser.add_argument(
        "--ai-model", help="AI model to use (e.g., gpt-4, gemini-1.5-pro)"
    )

    args = parser.parse_args()

    try:
        print("Enter your query (e.g., 'Create a Jira ticket for this feature.'):")
        user_query = input().strip()

        if not user_query:
            print("Error: No query provided.", file=sys.stderr)
            sys.exit(1)

        pr_number_str = args.pr_url.rstrip("/").split("/")[-1]
        print(f"Fetching details for pull request #{pr_number_str}...")

        pr_fetcher = PullRequestFetcher(args.github_token)
        pr_details = pr_fetcher.fetch_pr_details(args.pr_url)
        pr_context = format_pr_context(pr_details)

        print("Generating JIRA ticket using AI...")

        ai_client = AIClient(
            endpoint_url=args.ai_endpoint,
            api_key=args.ai_token,
            provider=args.ai_provider,
            ai_model=args.ai_model,
        )
        jira_ticket = ai_client.generate_jira_ticket(user_query, pr_context)

        print("\n" + "=" * 50)
        print("GENERATED JIRA TICKET:")
        print("=" * 50)
        print(jira_ticket)

    except requests.exceptions.HTTPError as e:
        print(
            f"Error: API request failed with status {e.response.status_code}:\n{e.response.text}",
            file=sys.stderr,
        )
        sys.exit(1)
    except (ValueError, IOError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
