"""Inference-time debiasing baselines used in the main text (paper Sec. 4.1).

The four starred (representative) baselines of the paper:

``vanilla``   -- no mitigation; the bias lower bound.
``sdd``       -- Self-Debiased Decoding, anti-prompt logit subtraction
                 (Schick et al., 2021).
``dexperts``  -- GeDi / DExperts-style expert-vs-anti-expert logit reweighting
                 toward neutral labels (Liu et al., 2021; Krause et al., 2021).
``dtcd``      -- Dual-Threshold Conformal Decoding: single-branch conformal
                 acceptance on toxicity and minimum probability.  This is the
                 closest baseline to COFT's CP component *without* counterfactual
                 reasoning.

All of them share :class:`coft.decoding.BaseDecoder`, so they run under the same
nucleus/temperature policy as COFT (App. C.2 "Fair decoding").
"""

from coft.baselines.dexperts import DExpertsDecoder
from coft.baselines.dtcd import DTCDDecoder
from coft.baselines.sdd import SDDDecoder
from coft.baselines.vanilla import VanillaDecoder

__all__ = ["VanillaDecoder", "SDDDecoder", "DExpertsDecoder", "DTCDDecoder"]
