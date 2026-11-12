"""iDOT protocol files (.idotx) from a pipetting sheet.

``idot_builder.py`` and ``iDOT_template.idotx`` are unmodified copies from
https://github.com/msoufi1453/idot-protocol-builder (commit d477d1e, 2026-10-20). Keep
them identical to that repository: to update, copy both files again. The template is
stored byte for byte (UTF-8 with a byte-order mark, CRLF line endings; see .gitattributes).

``metis.campaign.pipetting`` turns a round of recipes into the pipetting sheet and calls
the builder; use the builder directly for sheets written by hand.
"""
from . import idot_builder
from .idot_builder import IdotError

__all__ = ['idot_builder', 'IdotError']
