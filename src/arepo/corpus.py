"""Built-in passage corpus for the web demo and smoke tests."""

from copy import deepcopy


DEMO_CORPUS = [
    {
        "id": "declaration",
        "title": "Declaration of Independence",
        "source": "Continental Congress, 1776",
        "expected": "human",
        "kind": "public-domain classic",
        "passages": [
            {
                "title": "Preamble",
                "text": (
                    "When in the Course of human events, it becomes necessary for one people to dissolve "
                    "the political bands which have connected them with another, and to assume among the "
                    "powers of the earth, the separate and equal station to which the Laws of Nature and "
                    "of Nature's God entitle them, a decent respect to the opinions of mankind requires "
                    "that they should declare the causes which impel them to the separation."
                ),
            },
            {
                "title": "Self-evident truths",
                "text": (
                    "We hold these truths to be self-evident, that all men are created equal, that they "
                    "are endowed by their Creator with certain unalienable Rights, that among these are "
                    "Life, Liberty and the pursuit of Happiness. That to secure these rights, Governments "
                    "are instituted among Men, deriving their just powers from the consent of the governed."
                ),
            },
            {
                "title": "Mutual pledge",
                "text": (
                    "And for the support of this Declaration, with a firm reliance on the protection of "
                    "divine Providence, we mutually pledge to each other our Lives, our Fortunes and our "
                    "sacred Honor."
                ),
            },
        ],
    },
    {
        "id": "moby-dick",
        "title": "Moby-Dick",
        "source": "Herman Melville, 1851",
        "expected": "human",
        "kind": "public-domain novel",
        "passages": [
            {
                "title": "Loomings",
                "text": (
                    "Call me Ishmael. Some years ago, never mind how long precisely, having little or no "
                    "money in my purse, and nothing particular to interest me on shore, I thought I would "
                    "sail about a little and see the watery part of the world. It is a way I have of driving "
                    "off the spleen and regulating the circulation."
                ),
            },
            {
                "title": "The city",
                "text": (
                    "There now is your insular city of the Manhattoes, belted round by wharves as Indian "
                    "isles by coral reefs, commerce surrounds it with her surf. Right and left, the streets "
                    "take you waterward. Its extreme downtown is the battery, where that noble mole is "
                    "washed by waves, and cooled by breezes."
                ),
            },
            {
                "title": "November in the soul",
                "text": (
                    "Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly "
                    "November in my soul; whenever I find myself involuntarily pausing before coffin "
                    "warehouses, and bringing up the rear of every funeral I meet; and especially whenever "
                    "my hypos get such an upper hand of me, that it requires a strong moral principle to "
                    "prevent me from deliberately stepping into the street, and methodically knocking "
                    "people's hats off, then, I account it high time to get to sea as soon as I can."
                ),
            },
            {
                "title": "The watery world",
                "text": (
                    "Say you are in the country; in some high land of lakes. Take almost any path you "
                    "please, and ten to one it carries you down in a dale, and leaves you there by a pool "
                    "in the stream. There is magic in it. Let the most absent-minded of men be plunged in "
                    "his deepest reveries, stand that man on his legs, set his feet a-going, and he will "
                    "infallibly lead you to water, if water there be in all that region."
                ),
            },
        ],
    },
    {
        "id": "jane-eyre",
        "title": "Jane Eyre",
        "source": "Charlotte Bronte, 1847",
        "expected": "human",
        "kind": "public-domain novel",
        "passages": [
            {
                "title": "Opening weather",
                "text": (
                    "There was no possibility of taking a walk that day. We had been wandering, indeed, "
                    "in the leafless shrubbery an hour in the morning; but since dinner the cold winter "
                    "wind had brought with it clouds so sombre, and a rain so penetrating, that further "
                    "out-door exercise was now out of the question."
                ),
            },
            {
                "title": "Window seat",
                "text": (
                    "I mounted into the window-seat, gathering up my feet, and having drawn the red "
                    "moreen curtain nearly close, I was shrined in double retirement. Folds of scarlet "
                    "drapery shut in my view to the right hand; to the left were the clear panes of glass, "
                    "protecting, but not separating me from the drear November day."
                ),
            },
            {
                "title": "The red-room",
                "text": (
                    "The red-room was a square chamber, very seldom slept in, I might say never, indeed, "
                    "unless when a chance influx of visitors at Gateshead Hall rendered it necessary to "
                    "turn to account all the accommodation it contained. A bed supported on massive pillars "
                    "of mahogany, hung with curtains of deep red damask, stood out like a tabernacle in the centre."
                ),
            },
            {
                "title": "Lowood",
                "text": (
                    "I discovered, too, that a great pleasure, an enjoyment which the horizon only bounded, "
                    "lay all outside the high and spike-guarded walls of our garden. This pleasure consisted "
                    "in prospect of noble summits girdling a great hill-hollow, rich in verdure and shadow, "
                    "and in a bright beck, full of dark stones and sparkling eddies."
                ),
            },
        ],
    },
    {
        "id": "llm-ambiguous",
        "title": "LLM Control: Soft/Ambiguous",
        "source": "Synthetic control text generated for this repo",
        "expected": "ai",
        "kind": "synthetic AI-like text, weak local evidence",
        "passages": [
            {
                "title": "Policy explainer",
                "text": (
                    "A resilient city transportation plan should balance reliability, affordability, and "
                    "environmental impact. First, leaders can expand frequent bus corridors so residents "
                    "have predictable service throughout the day. Second, agencies can modernize fare "
                    "systems while preserving discounted passes for students, workers, and seniors."
                ),
            },
            {
                "title": "Productivity advice",
                "text": (
                    "Improving personal productivity often starts with reducing friction around the next "
                    "action. A useful approach is to define the outcome, break it into concrete steps, "
                    "schedule the first step, and review progress at the end of the week. This process "
                    "creates clarity without requiring an elaborate system."
                ),
            },
            {
                "title": "Hiring rubric",
                "text": (
                    "A fair hiring rubric should define the skills required for the role before interviews "
                    "begin. Reviewers can score examples of past work, structured problem solving, and "
                    "communication separately, then compare notes against the same criteria. This reduces "
                    "noise and helps the team distinguish evidence from impressions."
                ),
            },
            {
                "title": "Study plan",
                "text": (
                    "An effective study plan combines retrieval practice, spaced review, and short feedback "
                    "loops. Instead of rereading the same chapter repeatedly, a learner can answer targeted "
                    "questions, identify weak areas, and return to those concepts after a delay. The method "
                    "works best when progress is measured with concrete tasks."
                ),
            },
        ],
    },
    {
        "id": "llm-hard",
        "title": "LLM Control: Hard Evidence",
        "source": "HC3 ChatGPT answer",
        "expected": "ai",
        "kind": "AI-like text, hard local evidence",
        "passages": [
            {
                "title": "Bestseller explainer",
                "text": (
                    "There are many different best seller lists that are published by various organizations, "
                    "and the New York Times is just one of them. The New York Times best seller list is a "
                    "weekly list that ranks the best-selling books in the United States based on sales data "
                    "from a number of different retailers. The list is published in the New York Times "
                    "newspaper and is widely considered to be one of the most influential best seller lists "
                    "in the book industry. It is important to note that the New York Times best seller list "
                    "is not the only best seller list out there, and there are many other lists that rank "
                    "the top-selling books in different categories or in different countries. So it is "
                    "possible that a book could be a best seller on one list but not on another. Additionally, "
                    "the term best seller is often used more broadly to refer to any book that is selling "
                    "well, regardless of whether it is on a specific best seller list or not. So it is "
                    "possible that you may hear about a book being a best seller even if it is not "
                    "specifically ranked as a number one best seller on the New York Times list or any "
                    "other list."
                ),
            },
        ],
    },
]


def get_demo_corpus():
    """Return a defensive copy of the built-in passage corpus."""
    return deepcopy(DEMO_CORPUS)
