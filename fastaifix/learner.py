from fastai import *
from fastai.torch_core import *
from fastai.callback.all import *
from fastai.basics import *


def _print_shapes(o, bs):
#     if isinstance(o, torch.Size): return ' x '.join([str(bs)] + [str(t) for t in o[1:]])
#     else: return str([_print_shapes(x, bs) for x in o])
    return ' x '.join([str(bs)] + [str(t) for t in o[1:]])
    
def module_summary(learn, *xb):
    "Print a summary of `model` using `xb`"
    #Individual parameters wrapped in ParameterModule aren't called through the hooks in `layer_info`,
    #  thus are not counted inside the summary
    #TODO: find a way to have them counted in param n/mber somehow
    infos = layer_info(learn, *xb)
    n,bs = 64,find_bs(xb)
    inp_sz = _print_shapes(apply(lambda x:x.shape, xb), bs)
    res = f"{learn.model.__class__.__name__} (Input shape: {inp_sz})\n"
    res += "=" * n + "\n"
    res += f"{'Layer (type)':<20} {'Output Shape':<20} {'Param #':<10} {'Trainable':<10}\n"
    res += "=" * n + "\n"
    ps,trn_ps = 0,0
    infos = [o for o in infos if o is not None] #see comment in previous cell
    for typ,np,trn,sz in infos:
        if sz is None: continue
        ps += np
        if trn: trn_ps += np
        res += f"{typ:<20} {_print_shapes(sz, bs)[:19]:<20} {np:<10,} {str(trn):<10}\n"
        res += "_" * n + "\n"
    res += f"\nTotal params: {ps:,}\n"
    res += f"Total trainable params: {trn_ps:,}\n"
    res += f"Total non-trainable params: {ps - trn_ps:,}\n\n"
    return PrettyString(res)

def summary(learner):
    "Print a summary of the model, optimizer and loss function."
    xb = learner.dls.train.one_batch()[:learner.dls.train.n_inp]
    res = module_summary(learner, *xb)
    res += f"Optimizer used: {learner.opt_func}\nLoss function: {learner.loss_func}\n\n"
    if learner.opt is not None:
        res += f"Model " + ("unfrozen\n\n" if learner.opt.frozen_idx==0 else f"frozen up to parameter group #{learner.opt.frozen_idx}\n\n")
    res += "Callbacks:\n" + '\n'.join(f"  - {cb}" for cb in sort_by_run(learner.cbs))
    return PrettyString(res)

        