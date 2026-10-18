"""Step-by-step setup guides shown in the "Unlock cloud quality" wizard.

Every link was checked on 2026-10-17. Dashboard pages (fal.ai/dashboard/..., replicate.com/account/...)
send people to sign in first and then on to the page. Magnific's site blocks automated checks; its
API Keys link is the one Magnific's own quickstart gives (https://docs.magnific.com/quickstart).
Prices come from each provider's pricing page; see the notes at the top of fal.py, replicate.py
and magnific.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuideStep:
    title: str
    body: str
    link_label: str | None = None
    link_url: str | None = None


@dataclass(frozen=True)
class Guide:
    provider_id: str
    tagline: str
    best_for: str
    cost_note: str
    privacy_note: str
    key_hint: str
    steps: list[GuideStep]


_KEEP_IT_PRIVATE = "Keep it private, like a password: anyone who has it can spend your credit."

FAL = Guide(
    provider_id="fal",
    tagline="One key for four top upscalers, including SeedVR2 at full 8K.",
    best_for="The most choice for the money. SeedVR2 (the default) makes a photo-real 8K picture for about 5¢, "
    "and Topaz Precision keeps every detail exactly as it is.",
    cost_note="Pay as you go, no subscription. For an 8K picture made from a phone photo: SeedVR2 about 5¢, "
    "Topaz Precision about 24¢, Topaz Wonder about 56¢, Crystal about 81¢. Wider pictures cost a little less, "
    "and 16K costs about four times as much.",
    privacy_note="Your picture is sent to fal.ai to be upscaled. The app asks fal.ai not to keep it in your request "
    "history and to delete the finished picture after an hour. fal.ai's terms say your pictures stay yours.",
    key_hint="A long code, usually with a colon (:) in the middle. Copy all of it.",
    steps=[
        GuideStep(
            title="Make a fal.ai account",
            body="Open fal.ai and sign up (or log in if you already have an account).",
            link_label="Open fal.ai",
            link_url="https://fal.ai/login",
        ),
        GuideStep(
            title="Add a little credit",
            body="Open Billing and add credit with your card. $10 is plenty to start: that's dozens of 8K pictures. "
            "You only pay for the pictures you upscale.",
            link_label="Open fal.ai Billing",
            link_url="https://fal.ai/dashboard/billing",
        ),
        GuideStep(
            title="Create a key",
            body="Open API Keys and add a new key. Name it “Upscaler” and choose the API option when it asks "
            "what the key is for.",
            link_label="Open fal.ai API Keys",
            link_url="https://fal.ai/dashboard/keys",
        ),
        GuideStep(
            title="Copy the key and paste it here",
            body=f"Copy the whole key, paste it in the box below and click Test connection. {_KEEP_IT_PRIVATE}",
        ),
    ],
)

REPLICATE = Guide(
    provider_id="replicate",
    tagline="Topaz's most faithful upscaler, and your pictures are deleted after an hour.",
    best_for="Keeping faces, furniture and logos exactly as they are, with the strongest privacy.",
    cost_note="Pay as you go, no subscription: about 15¢ for an 8K picture made from a phone photo with Topaz "
    "High Fidelity, and 10–25¢ depending on the picture's size. 16K costs several times as much. "
    "Replicate doesn't charge for jobs that fail.",
    privacy_note="Your picture is sent to Replicate. Replicate deletes it, and the result, after one hour, and the "
    "app removes its upload as soon as your picture is done.",
    key_hint="Starts with r8_ followed by a long code.",
    steps=[
        GuideStep(
            title="Make a Replicate account",
            body="Open Replicate and sign in. If you're new, it walks you through making an account.",
            link_label="Open Replicate",
            link_url="https://replicate.com/signin",
        ),
        GuideStep(
            title="Add a card and some credit",
            body="Open Billing, add a payment card, then click Add credit. $10 is plenty to start: "
            "that's about 40 to 100 pictures at 8K, depending on their size.",
            link_label="Open Replicate Billing",
            link_url="https://replicate.com/account/billing",
        ),
        GuideStep(
            title="Create a token",
            body="Open API tokens, type a name like “Upscaler” and create the token.",
            link_label="Open Replicate API tokens",
            link_url="https://replicate.com/account/api-tokens",
        ),
        GuideStep(
            title="Copy the token and paste it here",
            body="Copy the whole token (it starts with r8_), paste it in the box below and click Test connection. "
            + _KEEP_IT_PRIVATE,
        ),
    ],
)

MAGNIFIC = Guide(
    provider_id="magnific",
    tagline="For people who already pay for Magnific (formerly Freepik).",
    best_for="Magnific's own look: crisp detail with Precision, or lots of new, imaginative detail with Creative.",
    cost_note="There's no price per picture: every upscale spends credits from your Magnific plan, even if your "
    "plan says “unlimited” on the website. Magnific doesn't publish what an 8K Precision picture costs. "
    "Creative costs about €0.50 for a 20-megapixel picture and stops at about 25 megapixels (around 6K). "
    "Magnific can't stop a picture once it has started, so cancelling still uses its credits.",
    privacy_note="Your picture is sent to Magnific. Links to finished pictures stop working after 30 minutes. "
    "Magnific doesn't say how long it keeps the original.",
    key_hint="A long mix of letters and numbers.",
    steps=[
        GuideStep(
            title="Check that your plan includes the API",
            body="Your web subscription doesn't cover other apps on its own: its “unlimited” upscales only work on "
            "the Magnific website. Apps like this one use Magnific's API, which always spends your plan's credits. "
            "Magnific has changed this a few times. Right now its pricing page lists API access on Premium and "
            "higher plans, and the Business plan is the one built for API use. If your account has no API Keys "
            "page, your plan doesn't include it.",
            link_label="See Magnific plans",
            link_url="https://www.magnific.com/pricing",
        ),
        GuideStep(
            title="Create an API key",
            body="Sign in with the account owner's email, open API Keys and create a new key.",
            link_label="Open Magnific API Keys",
            link_url="https://www.magnific.com/user/organization/api-keys",
        ),
        GuideStep(
            title="Copy the key and paste it here",
            body="Copy the whole key, paste it in the box below and click Test connection. "
            "Keep it private, like a password: anyone who has it can spend your credits.",
        ),
        GuideStep(
            title="No API on your plan?",
            body="Runway sells Magnific Precision pay as you go (about $1.50 per 8K picture), but this app can't "
            "connect to Runway yet. For faithful results that work in this app today, connect fal.ai instead: "
            "its Topaz Precision costs about 16–24¢ per 8K picture.",
        ),
    ],
)

GUIDES: dict[str, Guide] = {guide.provider_id: guide for guide in (FAL, REPLICATE, MAGNIFIC)}
