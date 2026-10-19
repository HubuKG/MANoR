from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .Trainer import Trainer
from .Tester import Tester

from .WCGTrainerGP import WCGTrainerGP
from .WCGTrainerTIVAGP import WCGTrainerTIVAGP


__all__ = [
	'Trainer',
	'Tester',
    'WCGTrainerGP',
    'WCGTrainerTIVAGP'
]
