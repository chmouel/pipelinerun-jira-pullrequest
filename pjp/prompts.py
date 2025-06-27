USER_PROMPT_TEMPLATE = """
Your task is to create a Jira issue. First, generate a suitable title for the
issue on its own line. Then, using the provided context, fill out the Jira
template below. The final output must be in Jira's text formatting, with the
title on the first line, followed by the filled template.

If you see in the github comment some directives like "/jira" make sure to understand
the query so it drives the ticket creation. Stil abiout the context of the
current pull request.

Add this pull request as a link in the description, make sure it's a jira text
formatted link, not markdown link.

Pull Request: {pr_url}

**User's Request:** "{user_query}"

**Pull Request Context and Discussions:**
{pr_context}

**Jira Template to Fill:**
{jira_template}

Please now generate the complete Jira issue, starting with the title on the very first line.
"""

JIRA_TEMPLATE = """h1. Story (Required)

As a <PERSONA> trying to <ACTION> I want <THIS OUTCOME>

_<Describes high level purpose and goal for this story. Answers the questions: Who is impacted, what is it and why do we need it? How does it improve the customer’s experience?>_
h2. *Background (Required)*

_<Describes the context or background related to this story>_
h2. *Out of scope*

_<Defines what is not included in this story>_
h2. *Approach (Required)*

_<Description of the general technical path on how to achieve the goal of the story. Include details like json schema, class definitions>_
h2. *Dependencies*

_<Describes what this story depends on. Dependent Stories and EPICs should be linked to the story.>_

 
h2. *Acceptance Criteria (Mandatory)*

_<Describe edge cases to consider when implementing the story and defining tests>_

_<Provides a required and minimum list of acceptance tests for this story. More is expected as the engineer implements this story>_

 
h1. *INVEST Checklist*

 Dependencies identified

 Blockers noted and expected delivery timelines set

 Design is implementable

 Acceptance criteria agreed upon

 Story estimated
h4. *Legend*

 Unknown

 Verified

 Unsatisfied

 
h2. *Done Checklist*
 * Code is completed, reviewed, documented and checked in
 * Unit and integration test automation have been delivered and running cleanly in continuous integration/staging/canary environment
 * Continuous Delivery pipeline(s) is able to proceed with new code included
 * Customer facing documentation, API docs etc. are produced/updated, reviewed and published
 * Acceptance criteria are met

h2. *Originial Pull Request number*

h3. *Original Pull Request Description*

"""
