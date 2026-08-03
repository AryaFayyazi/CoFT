"""Sensitive-span acquisition (paper Sec. 3.2 and App. D.2).

Spans reach the mask operator by two routes:

1. **User-specified lists.** Most bias benchmarks ship the protected term with
   the item (StereoSet's ``target``, BBQ's answer entities, BOLD's category),
   so the span is known exactly.  User lists take precedence (App. D.2, "When
   User Spans Disagree With NER").
2. **Detection.** A curated lexicon of protected-attribute markers plus, when
   spaCy is installed, an NER pass over ``PERSON`` / ``NORP`` (nationalities,
   religious and political groups).

Detected spans are unioned with user lists and overlapping spans are merged, so
``M`` stays idempotent and order-preserving.

The lexicon deliberately covers *localised, explicit* markers -- pronouns,
gendered role nouns, racial / ethnic / national / religious identifiers.  As the
paper states, COFT "is not itself a universal detector of all implicit bias"
(Sec. 3.2); it is a decode-time controller conditioned on aligned masking.
"""

from __future__ import annotations

import functools
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Set

__all__ = ["SensitiveLexicon", "LEXICON", "detect_spans", "spacy_entities"]


GENDER_TERMS: Sequence[str] = (
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "male", "males", "female", "females", "gentleman", "gentlemen", "lady", "ladies",
    "father", "dad", "mother", "mom", "son", "daughter", "brother", "sister",
    "husband", "wife", "boyfriend", "girlfriend", "uncle", "aunt",
    "grandfather", "grandmother", "grandpa", "grandma", "nephew", "niece",
    "mr", "mrs", "ms", "miss", "sir", "madam",
    "businessman", "businesswoman", "chairman", "chairwoman",
    "waiter", "waitress", "actor", "actress", "salesman", "saleswoman",
    "policeman", "policewoman", "widow", "widower", "bride", "groom",
    "transgender", "cisgender", "nonbinary", "non-binary", "gay", "lesbian",
    "bisexual", "queer", "straight", "homosexual", "heterosexual",
)

RACE_TERMS: Sequence[str] = (
    "black", "white", "asian", "latino", "latina", "latinx", "hispanic",
    "african", "african-american", "african american", "caucasian",
    "arab", "arabic", "middle eastern", "native american", "indigenous",
    "chinese", "japanese", "korean", "vietnamese", "indian", "pakistani",
    "bangladeshi", "filipino", "thai", "mexican", "colombian", "cuban",
    "puerto rican", "dominican", "brazilian", "nigerian", "ethiopian",
    "somali", "kenyan", "ghanaian", "moroccan", "egyptian", "lebanese",
    "syrian", "iraqi", "iranian", "afghan", "turkish", "russian", "ukrainian",
    "polish", "italian", "irish", "german", "french", "british", "english",
    "european", "african american", "jewish", "romani", "gypsy",
    "eskimo", "inuit", "aboriginal", "pacific islander", "samoan",
)

RELIGION_TERMS: Sequence[str] = (
    "muslim", "islamic", "islam", "christian", "christianity", "catholic",
    "protestant", "evangelical", "jewish", "judaism", "jew", "jews",
    "hindu", "hinduism", "buddhist", "buddhism", "sikh", "sikhism",
    "atheist", "atheism", "agnostic", "mormon", "orthodox", "baptist",
    "methodist", "lutheran", "quaker", "shia", "sunni", "rabbi", "imam",
    "priest", "pastor", "monk", "nun",
)

AGE_TERMS: Sequence[str] = (
    "young", "younger", "old", "older", "elderly", "aged", "senior", "seniors",
    "teenager", "teenage", "teen", "adolescent", "millennial", "boomer",
    "retiree", "retired", "middle-aged", "youngster", "grandparent",
)

NATIONALITY_TERMS: Sequence[str] = (
    "immigrant", "immigrants", "migrant", "refugee", "refugees", "foreigner",
    "foreign", "native", "citizen", "noncitizen", "undocumented", "expat",
)

DISABILITY_TERMS: Sequence[str] = (
    "disabled", "disability", "blind", "deaf", "mute", "wheelchair",
    "autistic", "autism", "adhd", "bipolar", "schizophrenic", "depressed",
    "handicapped", "paraplegic", "amputee", "dyslexic",
)

SES_TERMS: Sequence[str] = (
    "poor", "wealthy", "rich", "homeless", "welfare", "unemployed",
    "working-class", "upper-class", "lower-class", "affluent", "impoverished",
    "uneducated", "illiterate",
)

PHYSICAL_TERMS: Sequence[str] = (
    "fat", "obese", "overweight", "thin", "skinny", "tall", "short",
    "ugly", "beautiful", "attractive", "handsome",
)


#: Category -> terms.  Selecting a subset lets an experiment mask, say, only
#: gender markers (the ``COFT-Single`` variant of App. D.4).
LEXICON: Dict[str, Sequence[str]] = {
    "gender": GENDER_TERMS,
    "race": RACE_TERMS,
    "religion": RELIGION_TERMS,
    "age": AGE_TERMS,
    "nationality": NATIONALITY_TERMS,
    "disability": DISABILITY_TERMS,
    "socioeconomic": SES_TERMS,
    "physical": PHYSICAL_TERMS,
}


class SensitiveLexicon:
    """Lookup over the protected-attribute lexicon.

    Parameters
    ----------
    categories:
        Which families ``S_1 .. S_K`` to include.  ``None`` means all of them,
        which corresponds to the *joint mask* ``M_joint`` of App. D.4.
    extra:
        Additional user-supplied terms, unioned in.
    """

    def __init__(
        self,
        categories: Optional[Iterable[str]] = None,
        extra: Optional[Iterable[str]] = None,
    ) -> None:
        cats = list(categories) if categories is not None else list(LEXICON)
        unknown = [c for c in cats if c not in LEXICON]
        if unknown:
            raise ValueError(f"unknown lexicon categories {unknown}; available: {sorted(LEXICON)}")
        self.categories = cats
        terms: Set[str] = set()
        for c in cats:
            terms.update(t.lower() for t in LEXICON[c])
        if extra:
            terms.update(t.lower() for t in extra)
        # longest first so that multiword spans win over their constituents
        self.terms: List[str] = sorted(terms, key=lambda t: (-len(t), t))

    def __len__(self) -> int:
        return len(self.terms)

    def terms_in(self, text: str) -> List[str]:
        """Lexicon entries that occur in ``text`` (cheap pre-filter for the masker)."""
        low = text.lower()
        return [t for t in self.terms if t in low]


@functools.lru_cache(maxsize=1)
def _load_spacy(model: str = "en_core_web_sm"):
    try:
        import spacy
    except ImportError:  # pragma: no cover
        warnings.warn(
            "NER span detection was requested but spaCy is not installed; "
            "falling back to lexicon-only spans (pip install spacy && "
            "python -m spacy download en_core_web_sm).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    try:
        return spacy.load(model, disable=["lemmatizer", "tagger", "parser"])
    except Exception:  # pragma: no cover - model not downloaded
        warnings.warn(
            f"spaCy is installed but '{model}' is missing; falling back to "
            f"lexicon-only spans (python -m spacy download {model}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def spacy_entities(text: str, labels: Sequence[str] = ("PERSON", "NORP", "GPE")) -> List[str]:
    """Surface forms of protected-category entities, or ``[]`` if spaCy is unavailable.

    Optional by design: the paper's main results use lexicon + dataset-provided
    spans, and NER is the documented extension route (App. D.2).
    """
    nlp = _load_spacy()
    if nlp is None:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in labels]


def detect_spans(
    text: str,
    lexicon: Optional[SensitiveLexicon] = None,
    user_terms: Sequence[str] = (),
    use_ner: bool = False,
) -> List[str]:
    """Union of user terms, lexicon hits and (optionally) NER entities.

    Returns *surface strings*; :meth:`coft.masking.Masker.mask` turns them into
    character spans and then into exact token ranges.
    """
    lexicon = lexicon or SensitiveLexicon()
    found: List[str] = [t for t in user_terms if t]
    found.extend(lexicon.terms_in(text))
    if use_ner:
        found.extend(spacy_entities(text))
    # de-duplicate, keep longest-first ordering
    seen: Set[str] = set()
    out: List[str] = []
    for t in sorted(found, key=lambda s: (-len(s), s.lower())):
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out
