# pylint: disable=too-many-lines,too-many-branches,too-many-statements,too-many-locals,too-few-public-methods
"""
Script to create JIRA tickets from pull requests using AI assistance and the Jira API.
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import requests

from .prompts import JIRA_TEMPLATE, USER_PROMPT_TEMPLATE


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
        self.provider = provider or os.getenv("AI_PROVIDER", "openai")
        if self.provider is not None:
            self.provider = self.provider.lower()

        if endpoint_url:
            self.endpoint_url = endpoint_url.rstrip("/")
        else:
            self.endpoint_url = self.PROVIDER_ENDPOINTS.get(
                self.provider or "openai", self.PROVIDER_ENDPOINTS["openai"]
            )

        if api_key:
            self.api_key = api_key
        elif self.provider == "gemini":
            self.api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("AI_TOKEN")) or ""
        else:
            self.api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("AI_TOKEN")) or ""

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

    def generate_jira_ticket(self, user_query: str, pr_context: str, pr_url) -> str:
        """Generate JIRA ticket content using the specified AI provider."""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            pr_url=pr_url,
            user_query=user_query,
            pr_context=pr_context,
            jira_template=JIRA_TEMPLATE,
        )
        system_instruction = "You are an expert assistant skilled at creating well-structured Jira issues from technical discussions and pull request data. You will be given a user request, pull request context, and a Jira template. Your job is to synthesize all the information to generate a complete and accurate Jira issue, including a title."

        if self.provider == "gemini":
            url = f"{self.endpoint_url}/{self.ai_model}:generateContent?key={self.api_key}"
            full_prompt = f"{system_instruction}\n\n{user_prompt}"
            payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        else:
            url = self.endpoint_url
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
            payload = {
                "model": self.ai_model or "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 2500,
                "temperature": 0.5,
            }
        response = self.session.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        try:
            if self.provider == "gemini":
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise IOError(f"Unexpected API response format: {result}") from e


class JiraClient:
    """Handles creating tickets in Jira."""

    def __init__(self, endpoint: str, api_token: str):
        self.endpoint = endpoint.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        self.session.headers.update({"Authorization": f"Bearer {api_token}"})

    def create_ticket(
        self,
        project_key: str,
        summary: str,
        description: str,
        issuetype: str,
        component: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new issue in Jira."""
        api_url = f"{self.endpoint}/rest/api/2/issue"

        fields = {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issuetype},
        }

        if component:
            fields["components"] = [{"name": component}]

        payload = {"fields": fields}

        response = self.session.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()


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


def parse_ai_output_for_jira(ai_content: str) -> Tuple[str, str]:
    """Parses the AI-generated string to separate the title from the description."""
    parts = ai_content.strip().split("\n", 1)
    if not parts:
        return "", ""
    title = parts[0].strip()
    description = parts[1].strip() if len(parts) > 1 else ""
    return title, description


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Create JIRA tickets from pull requests using AI"
    )
    # --- GitHub and AI Arguments ---
    parser.add_argument(
        "pr_url", help="Pull request URL (e.g., https://github.com/owner/repo/pull/123)"
    )
    parser.add_argument(
        "--github-token", help="GitHub API token (or set GITHUB_TOKEN env var)"
    )
    parser.add_argument(
        "--ai-provider",
        choices=["openai", "gemini"],
        help="AI provider (can also be set via AI_PROVIDER env var)",
    )
    parser.add_argument(
        "--ai-token",
        help="AI API token (can also be set via corresponding env vars)",
    )
    parser.add_argument(
        "--ai-endpoint",
        help="Custom AI API endpoint (overrides provider default)",
    )
    parser.add_argument("--ai-model", help="AI model to use")

    # --- Jira Integration Arguments ---
    parser.add_argument(
        "--create-in-jira",
        action="store_true",
        help="Flag to enable creating the ticket in Jira.",
    )
    parser.add_argument(
        "--jira-endpoint",
        help="Jira endpoint URL (e.g., https://your-domain.atlassian.net). Can be set via JIRA_ENDPOINT env var.",
    )
    parser.add_argument(
        "--jira-user-email",
        help="Jira user email. Can be set via JIRA_USER_EMAIL env var.",
    )
    parser.add_argument(
        "--jira-token",
        help="Jira API token. Can be set via JIRA_TOKEN env var.",
    )
    parser.add_argument(
        "--jira-project",
        help="Jira project key (e.g., 'PROJ'). Can be set via JIRA_PROJECT env var.",
    )
    parser.add_argument(
        "--jira-issuetype",
        default="Story",
        help="Jira issue type (default: 'Story'). Can be set via JIRA_ISSUETYPE env var.",
    )
    parser.add_argument(
        "--jira-component",
        help="Jira component name. Can be set via JIRA_COMPONENT env var.",
    )
    parser.add_argument(
        "--jira-text",
        help="Optional text to use as the Jira ticket content instead of AI-generated content.",
    )

    parser.add_argument(
        "--pr-query",
        help="Optional user query to guide AI generation. If not provided, will prompt for input.",
    )

    args = parser.parse_args()

    try:
        # --- Step 1: Get User Input and Fetch PR Data ---
        if not args.pr_query:
            print("Enter your query (e.g., 'Create a Jira ticket for this feature.'):")
            pr_query = input().strip()
            if not pr_query:
                print("Error: No query provided.", file=sys.stderr)
                sys.exit(1)
            args.pr_query = pr_query

        pr_number_str = args.pr_url.rstrip("/").split("/")[-1]
        print(f"Fetching details for pull request #{pr_number_str}...")
        pr_fetcher = PullRequestFetcher(args.github_token)
        pr_details = pr_fetcher.fetch_pr_details(args.pr_url)
        pr_context = format_pr_context(pr_details)

        # --- Step 2: Generate Ticket Content with AI ---
        print("Generating JIRA ticket content using AI...")
        ai_client = AIClient(
            endpoint_url=args.ai_endpoint,
            api_key=args.ai_token,
            provider=args.ai_provider,
            ai_model=args.ai_model,
        )
        if args.jira_text:
            if os.path.exists(args.jira_text):
                with open(args.jira_text, "r", encoding="utf-8") as file:
                    generated_content = file.read()
            else:
                generated_content = args.jira_text
        else:
            generated_content = ai_client.generate_jira_ticket(
                args.pr_query, pr_context, args.pr_url
            )

        print("\n" + "=" * 50)
        print("AI-GENERATED JIRA TICKET CONTENT:")
        print("=" * 50)
        print(generated_content)
        print("=" * 50 + "\n")

        # --- Step 3: Create Ticket in Jira (if requested) ---
        if not args.create_in_jira:
            print(
                "Jira ticket creation is disabled. Use --create-in-jira to enable it."
            )
            sys.exit(0)

        # Get Jira config from args or environment variables
        jira_endpoint = args.jira_endpoint or os.getenv("JIRA_ENDPOINT")
        jira_user_email = args.jira_user_email or os.getenv("JIRA_USER_EMAIL")
        jira_token = args.jira_token or os.getenv("JIRA_TOKEN")
        jira_project = args.jira_project or os.getenv("JIRA_PROJECT")
        jira_issuetype = args.jira_issuetype or os.getenv("JIRA_ISSUETYPE", "Story")
        jira_component = args.jira_component or os.getenv("JIRA_COMPONENT")

        required_jira_args = {
            "endpoint": jira_endpoint,
            "user email": jira_user_email,
            "token": jira_token,
            "project key": jira_project,
        }

        missing_args = [name for name, val in required_jira_args.items() if not val]
        if missing_args:
            print(
                f"Error: Missing required Jira arguments: {', '.join(missing_args)}",
                file=sys.stderr,
            )
            sys.exit(1)

        print("Proceeding with Jira ticket creation...")

        # Parse AI output
        title, description = parse_ai_output_for_jira(generated_content)
        if not title or not description:
            print(
                "Error: Could not parse title and description from AI output.",
                file=sys.stderr,
            )
            print(f"Generated content was:\n{generated_content}", file=sys.stderr)
            sys.exit(1)

        # Create Jira ticket
        jira_client = JiraClient(jira_endpoint, jira_token)
        created_ticket = jira_client.create_ticket(
            project_key=jira_project,
            summary=title,
            description=description,
            issuetype=jira_issuetype,
            component=jira_component,
        )

        ticket_key = created_ticket.get("key")
        ticket_url = f"{jira_endpoint.rstrip('/')}/browse/{ticket_key}"

        print("\n" + "*" * 50)
        print(f"Successfully created Jira ticket: {ticket_key}")
        print(f"URL: {ticket_url}")
        print("*" * 50)

        comments_url = f"https://api.github.com/repos/{pr_details['pr_data']['base']['repo']['owner']['login']}/{pr_details['pr_data']['base']['repo']['name']}/issues/{pr_details['pr_data']['number']}/comments"
        github_token = args.github_token or os.getenv("GITHUB_TOKEN")
        if github_token:
            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json",
            }
            comment_body = {
                "body": f"💫 Jira ticket created: [{ticket_key}]({ticket_url})"
            }
            response = requests.post(
                comments_url, json=comment_body, headers=headers, timeout=30
            )
            html_url = None
            if (
                response.status_code == 201
                and "comment" in response.json()
                and "html_url" in response.json()["comment"]
            ):
                html_url = response.json()["comment"]["html_url"]
            if response.ok:
                print("Posted Jira ticket URL as a comment on the GitHub PR.")
                if html_url:
                    print(f"Comment URL: {html_url}")
            else:
                print(
                    f"Failed to post Jira ticket URL to GitHub PR: {response.text}",
                    file=sys.stderr,
                )
        else:
            print("No GitHub token provided; skipping posting Jira ticket URL to PR.")

    except requests.exceptions.HTTPError as e:
        error_message = e.response.text
        print(
            f"Error: API request failed with status {e.response.status_code}:\n{error_message}",
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
