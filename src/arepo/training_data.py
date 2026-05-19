"""Training data loaders shared by MoE expert builders."""

import json
from pathlib import Path

from .download import (
    load_sample_texts,
    load_cgtd,
    load_hc3,
    load_wiki,
    load_raid,
    load_mage,
    load_arepo_essays,
)


HISTORIC_CIVIC_HUMAN = [
    (
        "When in the Course of human events, it becomes necessary for one people to dissolve "
        "the political bands which have connected them with another, and to assume among the "
        "powers of the earth, the separate and equal station to which the Laws of Nature and "
        "of Nature's God entitle them, a decent respect to the opinions of mankind requires "
        "that they should declare the causes which impel them to the separation."
    ),
    (
        "We hold these truths to be self-evident, that all men are created equal, that they "
        "are endowed by their Creator with certain unalienable Rights, that among these are "
        "Life, Liberty and the pursuit of Happiness. That to secure these rights, Governments "
        "are instituted among Men, deriving their just powers from the consent of the governed."
    ),
    (
        "And for the support of this Declaration, with a firm reliance on the protection of "
        "divine Providence, we mutually pledge to each other our Lives, our Fortunes and our "
        "sacred Honor."
    ),
    (
        "When in the Course of human events, it becomes necessary for one people to dissolve "
        "the political bands which have connected them with another, and to assume among the "
        "powers of the earth, the separate and equal station to which the Laws of Nature and "
        "of Nature's God entitle them, a decent respect to the opinions of mankind requires "
        "that they should declare the causes which impel them to the separation.\n\n"
        "We hold these truths to be self-evident, that all men are created equal, that they "
        "are endowed by their Creator with certain unalienable Rights, that among these are "
        "Life, Liberty and the pursuit of Happiness. That to secure these rights, Governments "
        "are instituted among Men, deriving their just powers from the consent of the governed.\n\n"
        "And for the support of this Declaration, with a firm reliance on the protection of "
        "divine Providence, we mutually pledge to each other our Lives, our Fortunes and our "
        "sacred Honor."
    ),
    (
        "Four score and seven years ago our fathers brought forth on this continent "
        "a new nation, conceived in Liberty, and dedicated to the proposition that "
        "all men are created equal. Now we are engaged in a great civil war, testing "
        "whether that nation, or any nation so conceived and so dedicated, can long endure."
    ),
    (
        "We the People of the United States, in Order to form a more perfect Union, "
        "establish Justice, insure domestic Tranquility, provide for the common defence, "
        "promote the general Welfare, and secure the Blessings of Liberty to ourselves "
        "and our Posterity, do ordain and establish this Constitution for the United States of America."
    ),
    (
        "After an unequivocal experience of the inefficacy of the subsisting federal "
        "government, you are called upon to deliberate on a new Constitution for the "
        "United States of America. The subject speaks its own importance; comprehending "
        "in its consequences nothing less than the existence of the Union."
    ),
    (
        "Friends and Fellow-Citizens: The period for a new election of a citizen to "
        "administer the executive government of the United States being not far distant, "
        "it appears to me proper, especially as it may conduce to a more distinct "
        "expression of the public voice, that I should now apprise you of the resolution I have formed."
    ),
    (
        "With malice toward none, with charity for all, with firmness in the right as "
        "God gives us to see the right, let us strive on to finish the work we are in, "
        "to bind up the nation's wounds, to care for him who shall have borne the battle "
        "and for his widow and his orphan."
    ),
    (
        "The powers delegated by the proposed Constitution to the federal government "
        "are few and defined. Those which are to remain in the State governments are "
        "numerous and indefinite. The former will be exercised principally on external "
        "objects, as war, peace, negotiation, and foreign commerce."
    ),
]


HISTORIC_CIVIC_AI = [
    (
        "A modern civic compact should begin by defining the rights of residents and "
        "the responsibilities of public institutions. First, the government should "
        "protect equal access to basic services. Second, it should ensure that budgets, "
        "records, and decisions are transparent enough for citizens to evaluate."
    ),
    (
        "An effective democratic framework balances representation, accountability, "
        "and practical administration. Leaders can strengthen public trust by publishing "
        "clear rules, explaining tradeoffs, and creating regular opportunities for "
        "community review before major policies are adopted."
    ),
    (
        "A resilient transportation policy should connect neighborhoods, reduce travel "
        "costs, and limit environmental harm. The city can expand reliable bus routes, "
        "coordinate schedules with schools and employers, and preserve discounted fares "
        "for students, workers, and seniors."
    ),
    (
        "Public safety planning works best when prevention, response, and accountability "
        "are considered together. Agencies should invest in emergency readiness, maintain "
        "clear reporting channels, and review outcomes so that future decisions are based "
        "on evidence rather than habit."
    ),
    (
        "A useful education policy should describe the desired outcome, the resources "
        "needed to reach it, and the method for measuring progress. Schools can support "
        "students by combining strong instruction, accessible counseling, and consistent "
        "communication with families."
    ),
    (
        "Improving personal productivity often starts with reducing friction around the "
        "next action. A practical system defines the goal, breaks it into concrete steps, "
        "schedules the first step, and reviews progress at the end of the week."
    ),
    (
        "A responsible climate plan should identify near-term actions and long-term "
        "maintenance costs. Officials can improve building efficiency, electrify public "
        "fleets, protect tree cover, and publish annual updates so residents can see "
        "whether emissions are actually falling."
    ),
    (
        "Good workplace policy is easiest to follow when expectations are explicit. "
        "Teams should define roles, document recurring decisions, create escalation paths, "
        "and revisit the process after major projects so improvements become part of the "
        "normal operating rhythm."
    ),
]


def add_dataset(records, name, human_texts, ai_texts):
    """Append labeled records from one dataset."""
    for text in human_texts:
        records.append({"text": text, "label": 0, "group": name})
    for text in ai_texts:
        records.append({"text": text, "label": 1, "group": name})


def load_historic_civic():
    """Load the local historic-civic calibration source."""
    data_file = Path(__file__).parent / "data" / "historic_civic.jsonl"
    if data_file.exists():
        human_texts = []
        ai_texts = []
        with data_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["label"] == "human":
                    human_texts.append(row["text"])
                elif row["label"] == "ai":
                    ai_texts.append(row["text"])
        return human_texts, ai_texts
    return list(HISTORIC_CIVIC_HUMAN), list(HISTORIC_CIVIC_AI)


def load_training_records(data, max_per_class):
    """Load training records with dataset provenance preserved."""
    records = []
    external_cap = 50 if max_per_class is None else max_per_class
    if data == "sample":
        human_texts, ai_texts = load_sample_texts()
        add_dataset(records, "sample", human_texts, ai_texts)
    elif data == "cgtd":
        human_texts, ai_texts = load_cgtd(max_per_class=max_per_class)
        add_dataset(records, "cgtd", human_texts, ai_texts)
    elif data == "hc3":
        human_texts, ai_texts = load_hc3(max_per_class=external_cap)
        add_dataset(records, "hc3", human_texts, ai_texts)
    elif data == "wiki":
        human_texts, ai_texts = load_wiki(max_per_class=external_cap)
        add_dataset(records, "wiki", human_texts, ai_texts)
    elif data == "raid":
        human_texts, ai_texts = load_raid(max_per_class=external_cap)
        add_dataset(records, "raid", human_texts, ai_texts)
    elif data == "mage":
        human_texts, ai_texts = load_mage(max_per_class=external_cap)
        add_dataset(records, "mage", human_texts, ai_texts)
    elif data == "arepo":
        human_texts, ai_texts = load_arepo_essays(max_per_class=external_cap)
        add_dataset(records, "arepo", human_texts, ai_texts)
    elif data == "historic_civic":
        human_texts, ai_texts = load_historic_civic()
        add_dataset(records, "historic_civic", human_texts, ai_texts)
    elif data == "combined":
        for name, loader in (
            ("cgtd", load_cgtd),
            ("hc3", load_hc3),
        ):
            if name == "cgtd":
                human_texts, ai_texts = loader()
            else:
                human_texts, ai_texts = loader(max_per_class=external_cap)
            add_dataset(records, name, human_texts, ai_texts)
            print(f"  {name}: {len(human_texts)} human, {len(ai_texts)} AI")
    elif data == "extended":
        human_texts, ai_texts = load_sample_texts()
        add_dataset(records, "sample", human_texts, ai_texts)
        print(f"  sample: {len(human_texts)} human, {len(ai_texts)} AI")
        loaders = [
            ("cgtd", load_cgtd),
            ("hc3", load_hc3),
            ("wiki", load_wiki),
            ("raid", load_raid),
            ("mage", load_mage),
            ("arepo", load_arepo_essays),
        ]
        for name, loader in loaders:
            human_texts, ai_texts = loader(max_per_class=external_cap)
            add_dataset(records, name, human_texts, ai_texts)
            print(f"  {name}: {len(human_texts)} human, {len(ai_texts)} AI")
    return records
