from models import cae1dm, sscnet, cae3d

from models import hycot

models = {
    # --- HySpecNet-11k BASELINES ---
    "cae1d_cr4": cae1dm.cae1d_cr4,
    "cae1d_cr8": cae1dm.cae1d_cr8,
    "cae1d_cr16": cae1dm.cae1d_cr16,
    "cae1d_cr32": cae1dm.cae1d_cr32,

    "sscnet_cr4": sscnet.sscnet_cr4,
    "sscnet_cr8": sscnet.sscnet_cr8,
    "sscnet_cr16": sscnet.sscnet_cr16,
    "sscnet_cr32": sscnet.sscnet_cr32,

    "cae3d_cr4": cae3d.cae3d_cr4,
    "cae3d_cr8": cae3d.cae3d_cr8,
    "cae3d_cr16": cae3d.cae3d_cr16,
    "cae3d_cr32": cae3d.cae3d_cr32,

    # --- HYCOT ---
    "hycot_cr4": hycot.hycot_cr4,
    "hycot_cr8": hycot.hycot_cr8,
    "hycot_cr16": hycot.hycot_cr16,
    "hycot_cr32": hycot.hycot_cr32,
}
