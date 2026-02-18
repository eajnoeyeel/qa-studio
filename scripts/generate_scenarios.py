#!/usr/bin/env python3
"""
Scenario generator for SaaS CS QA Studio.
Generates synthetic customer service tickets for a collaboration tool.
"""
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Taxonomy categories
CATEGORIES = [
    "billing_seats",
    "billing_refund",
    "workspace_access",
    "permission_sharing",
    "login_sso",
    "import_export_sync",
    "bug_report",
    "feature_request",
]

# Scenario templates per category
SCENARIOS = {
    "billing_seats": [
        {
            "user_messages": [
                "Hi, I need to add 5 more seats to our Team plan. How do I do that?",
                "We're onboarding new team members next week.",
            ],
            "candidate_responses": {
                "good": "Happy to help you add seats! Go to Settings → Billing → Manage Seats, and click 'Add Seats'. Enter 5 for the number of seats. The charge will be prorated for your current billing cycle. Would you like me to walk you through it step by step?",
                "bad_overclaim": "Our AI will automatically detect your team size and add seats for free! The system will handle everything in the next 24 hours.",
                "bad_pii": "Sure! First, could you send me your credit card number so I can process this immediately?",
                "bad_no_action": "Thanks for reaching out about seats. Seats are important for team collaboration.",
            }
        },
        {
            "user_messages": [
                "We want to remove some seats from our plan. We had a layoff and now have fewer people.",
                "Currently paying for 20 seats but only need 12.",
            ],
            "candidate_responses": {
                "good": "I understand, and I'm sorry to hear about the changes. You can reduce seats by going to Settings → Members, removing the members who no longer need access, and then adjusting your seat count in Settings → Billing. The reduction will take effect at the end of your current billing cycle. Would you like help with this process?",
                "bad_tone": "Whatever. Just go to billing and remove seats. It's not that complicated.",
            }
        },
    ],
    "billing_refund": [
        {
            "user_messages": [
                "I was charged twice this month! I see two charges of $80 on my statement.",
                "This is frustrating, I need this resolved ASAP.",
            ],
            "candidate_responses": {
                "good": "I sincerely apologize for this billing error. Duplicate charges should never happen. I can see you're rightfully frustrated. Let me look into this immediately. Could you please provide the last 4 digits of your payment card and the dates of both charges so I can locate the transactions and process a refund for the duplicate?",
                "bad_no_action": "I understand you're upset about the charges. Billing issues can be frustrating for everyone.",
            }
        },
        {
            "user_messages": [
                "I want a full refund. I subscribed 3 weeks ago but the product doesn't work for our use case.",
            ],
            "candidate_responses": {
                "good": "I'm sorry to hear the product hasn't met your needs. For subscriptions within the first 14 days, we offer full refunds. Since you're at 3 weeks, I'd recommend reaching out to our billing team who can review your case - sometimes exceptions are made. You can email billing@example.com with your account email and the reason for the refund request. Is there anything about the product I could help clarify that might change your experience?",
                "bad_overclaim": "Yes, we definitely will give you a full refund no matter how long you've been subscribed! I'll process that right now.",
            }
        },
    ],
    "workspace_access": [
        {
            "user_messages": [
                "I can't access our team workspace anymore. I was able to yesterday.",
                "Getting an error that says 'Workspace not found'.",
            ],
            "candidate_responses": {
                "good": "Let me help you troubleshoot this. A 'Workspace not found' error usually means one of a few things: 1) You might be logged into a different account than the one with workspace access, 2) Your access might have been changed by a workspace admin, or 3) There could be a temporary sync issue. First, can you confirm you're logged in with the correct email? You can check this by clicking your profile icon in the bottom left.",
                "bad_missing_info": "That error means you don't have access. Contact your admin.",
            }
        },
        {
            "user_messages": [
                "How do I create a new workspace for a different project?",
            ],
            "candidate_responses": {
                "good": "Creating a new workspace is simple! Click on your current workspace name in the sidebar, then select 'Create or Join Workspace' at the bottom. Choose 'Create a new workspace', give it a name, and select whether you want a free workspace or to add it to your existing paid plan. Would you like separate billing for this workspace or add it to your current subscription?",
                "bad_vague": "You can create workspaces from the sidebar menu.",
            }
        },
    ],
    "permission_sharing": [
        {
            "user_messages": [
                "I shared a page with my colleague but they say they can't edit it.",
                "I definitely shared it with edit access, I'm sure of it.",
            ],
            "candidate_responses": {
                "good": "Let's figure this out together. There are a few things to check: 1) On the page, click 'Share' and verify your colleague is listed with 'Can edit' permission, 2) Check if the page inherits permissions from a parent page that might be restricting access, 3) Ask your colleague to refresh their browser. Also, is your colleague a full workspace member or a guest? Guests sometimes have different default permissions.",
                "bad_permission_mismatch": "They probably need to be a workspace admin to edit. Make them an admin and it will work.",
            }
        },
        {
            "user_messages": [
                "How do I share a page publicly so anyone with the link can view it?",
            ],
            "candidate_responses": {
                "good": "To share a page publicly: 1) Open the page and click 'Share' in the top right, 2) Toggle on 'Share to web', 3) Copy the public link. Note: If you don't see this option, it might be disabled by your workspace admin. Would you like the link to allow comments or be completely view-only?",
                "bad_pii_risk": "Just send me the page link and I'll make it public for you from our admin panel.",
            }
        },
    ],
    "login_sso": [
        {
            "user_messages": [
                "I keep getting SAML_002 error when trying to log in with SSO.",
                "Our IT team says I'm set up correctly in Okta.",
            ],
            "candidate_responses": {
                "good": "SAML_002 means your account isn't properly provisioned on our end. Even though you're set up in Okta, the user needs to be provisioned to our app. Your IT admin needs to: 1) Check if SCIM provisioning is enabled, 2) If not, manually trigger a sync from Okta, or 3) Try removing and re-adding you in the Okta app assignment. Would you like me to provide documentation for your IT team?",
                "bad_sso_miss": "Just use your email and password to log in instead of SSO. It's easier anyway.",
            }
        },
        {
            "user_messages": [
                "We're setting up SSO with Azure AD. What do we need?",
                "We have an Enterprise plan.",
            ],
            "candidate_responses": {
                "good": "Great, Enterprise includes SSO! Here's what you'll need: 1) Admin access to Azure AD, 2) Workspace owner or admin role in our app, 3) Your company domain verified in our settings. The setup involves adding our app from the Azure AD gallery, configuring SAML with our metadata URL, and mapping user attributes. I can send you our detailed Azure AD integration guide - would that help?",
                "bad_overclaim": "SSO setup with Azure AD takes about 2 minutes and our system will auto-configure everything.",
            }
        },
    ],
    "import_export_sync": [
        {
            "user_messages": [
                "I'm trying to import from Notion but it keeps failing.",
                "I exported as Markdown and HTML both.",
            ],
            "candidate_responses": {
                "good": "Let's troubleshoot your Notion import. A few things to check: 1) File size - we support up to 50MB per file, 2) For Notion exports, the Markdown format usually works best, 3) Make sure to export 'Include subpages' and 'Include databases as CSV'. Could you tell me roughly how large your Notion export is, and if there are any specific error messages you're seeing?",
                "bad_no_steps": "Imports can fail for various reasons. Try again later or use a different format.",
            }
        },
        {
            "user_messages": [
                "How can I set up automatic sync with our Google Drive?",
            ],
            "candidate_responses": {
                "good": "We support Google Drive integration for linking files! Here's how to set it up: 1) Go to Settings → Integrations, 2) Click 'Connect' next to Google Drive, 3) Authorize access to your Drive. Once connected, you can embed Drive files in pages using /google-drive. Note: This is a link integration, not a two-way sync. Changes made in Drive reflect automatically, but we don't push changes back to Drive. Would you like me to explain more about how the integration works?",
                "bad_overclaim": "Our sync with Google Drive will automatically keep everything perfectly in sync in real-time with zero effort!",
            }
        },
    ],
    "bug_report": [
        {
            "user_messages": [
                "The app keeps crashing when I try to open a specific page.",
                "It happens every time, I've tried reloading multiple times.",
            ],
            "candidate_responses": {
                "good": "I'm sorry you're experiencing crashes - that shouldn't happen. Let me help troubleshoot: 1) Try accessing the page in incognito mode to rule out browser extensions, 2) Clear your browser cache, 3) Try a different browser. If it still crashes, the page might have corrupted content. Could you share the page URL (without any private content) so we can investigate? Also, what browser and operating system are you using?",
                "bad_dismissive": "Have you tried turning it off and on again? That usually fixes these things.",
            }
        },
        {
            "user_messages": [
                "Images aren't loading in any of my documents.",
                "Everything else works fine, just images show a broken icon.",
            ],
            "candidate_responses": {
                "good": "That's frustrating - let's figure this out. Images not loading could be due to: 1) Network/firewall blocking our image CDN, 2) Browser extension conflicts, 3) Cache issues. First, try: clearing cache, disabling ad blockers temporarily, and checking if images load on your phone or different network. Are you on a corporate network that might have strict firewall rules?",
                "bad_impossible_promise": "Our engineers will fix this in the next 30 minutes and all your images will be restored automatically.",
            }
        },
    ],
    "feature_request": [
        {
            "user_messages": [
                "I'd love to have dark mode in the mobile app.",
                "The web version has it but not mobile.",
            ],
            "candidate_responses": {
                "good": "Great feedback! You're right that dark mode is currently available on web but not our mobile apps. This is actually one of our most requested features. I'll make sure your vote is counted. In the meantime, some users have had success using their device's system-wide dark mode with accessibility settings. Would you like me to submit this formally to our product team with your use case?",
                "bad_false_timeline": "Dark mode for mobile will definitely be released next month! I guarantee it.",
            }
        },
        {
            "user_messages": [
                "Is there any way to have recurring tasks in the database?",
                "Like tasks that automatically create a new copy every week?",
            ],
            "candidate_responses": {
                "good": "Recurring tasks aren't built-in yet, but there are a couple workarounds: 1) You can use our API with a tool like Zapier or Make to automatically create tasks on a schedule, 2) Some users create template pages and duplicate them manually. I know it's not ideal - recurring tasks is a frequently requested feature. Would you like me to add your use case to our product feedback? What specifically would you need - weekly, monthly, custom schedules?",
                "bad_wrong_info": "Yes, we have recurring tasks! Just right-click on any task and select 'Make Recurring'.",
            }
        },
    ],
}


def generate_ticket(category: str, quality: str = "good") -> Dict[str, Any]:
    """Generate a single ticket."""
    scenario = random.choice(SCENARIOS[category])

    # Build conversation
    messages = []
    for msg in scenario["user_messages"]:
        messages.append({
            "role": "user",
            "content": msg,
        })

    # Select candidate response based on quality
    responses = scenario["candidate_responses"]
    if quality == "good":
        response_key = "good"
    elif quality == "mixed":
        response_key = random.choice(list(responses.keys()))
    else:  # bad
        bad_keys = [k for k in responses.keys() if k != "good"]
        response_key = random.choice(bad_keys) if bad_keys else "good"

    candidate_response = responses.get(response_key, responses["good"])

    return {
        "id": str(uuid.uuid4()),
        "conversation": messages,
        "candidate_response": candidate_response,
        "metadata": {
            "category": category,
            "response_quality": response_key,
            "generated_at": datetime.utcnow().isoformat(),
        }
    }


def generate_dataset(count: int = 50, split: str = "dev") -> List[Dict[str, Any]]:
    """Generate a dataset of tickets."""
    tickets = []

    # Distribute across categories
    per_category = count // len(CATEGORIES)
    remainder = count % len(CATEGORIES)

    for i, category in enumerate(CATEGORIES):
        num = per_category + (1 if i < remainder else 0)
        for _ in range(num):
            # Mix of quality levels
            quality = random.choices(
                ["good", "mixed", "bad"],
                weights=[0.5, 0.3, 0.2]
            )[0]
            ticket = generate_ticket(category, quality)
            tickets.append(ticket)

    random.shuffle(tickets)
    return tickets


def main():
    """Generate and save sample data."""
    import os

    # Ensure directory exists
    os.makedirs("sample_data", exist_ok=True)

    # Generate datasets
    datasets = {
        "dev": 30,
        "test": 15,
        "ab_eval": 10,
    }

    for split, count in datasets.items():
        tickets = generate_dataset(count, split)
        filename = f"sample_data/tickets_{split}.jsonl"

        with open(filename, "w") as f:
            for ticket in tickets:
                f.write(json.dumps(ticket) + "\n")

        print(f"Generated {len(tickets)} tickets for {split} split -> {filename}")

    print("\nTotal tickets generated:", sum(datasets.values()))


if __name__ == "__main__":
    main()
