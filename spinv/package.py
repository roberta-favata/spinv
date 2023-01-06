from . import _pythtb
from . import _tbmodels

import pythtb 
import tbmodels

def single_point_chern(model, formula='symmetric'):
    if isinstance(model, pythtb.tb_model):
        return _pythtb.single_point_chern(model, formula)
    elif isinstance(model, tbmodels.Model):
        return _tbmodels.single_point_chern(model, formula)
    else:
        raise NotImplementedError('Invalid model.')


def single_point_spin_chern(model, spin='down', formula='symmetric', return_gap_pszp=False):
    if isinstance(model, pythtb.tb_model):
        return _pythtb.single_point_spin_chern(model, spin, formula, return_gap_pszp)
    elif isinstance(model, tbmodels.Model):
        return _tbmodels.single_point_spin_chern(model, spin, formula, return_gap_pszp)
    else:
        raise NotImplementedError('Invalid model.')

