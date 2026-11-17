import numpy as np
import pynucastro as pyna
from pynucastro import Nucleus


def get_aprox19_comp(comp):
    aprox19_comp = [Nucleus("he3"), Nucleus("he4"), Nucleus("c12"), Nucleus("n14"),
                    Nucleus("o16"), Nucleus("ne20"), Nucleus("mg24"), Nucleus("si28"),
                    Nucleus("s32"), Nucleus("ar36"), Nucleus("ca40"), Nucleus("ti44"),
                    Nucleus("cr48"), Nucleus("fe52"), Nucleus("fe54"), Nucleus("ni56"),
                    Nucleus("n"), Nucleus("p")]
    reduced_comp = comp.bin_as(aprox19_comp, exclude=[Nucleus("ni56")])

    # the actual aprox19 network has 2 protons, one is the NSE proton
    # and the other is H that participates in H-burning.  We need to
    # to add H1 with zero abundance in our return

    # we also want to preserve the order that aprox19 uses, with H1 first, and n and p last
    X = {}
    X["H1"] = 0.0
    for nuc, comp in reduced_comp.items():
        X[nuc] = comp

    return X


def make_nse_network():

    # list of nuclei we care about

    nuc_list = ["n", "p", "d", "he3-4", "c12", "o16", "n13-14", "f18", "ne20", "ne21-22",
                "na23", "mg24", "si28", "s32", "ar36", "ca40", "sc43", "al27", "p31",
                "cl35", "k39", "ca45-48", "sc45-49", "ti44-52", "v47-54", "cr48-56", 
                "mn51-58", "fe52-60", "co54-61", "ni56-65", "cu59", "zn60"]

    rc = network_helper(nuc_list, use_detailed_balance=False, network_type="nse")

    return rc

def output_header(Ts, rhos, yes):

    with open("nse_table_size.H", "w") as nse_h:

        nse_h.write("#ifndef NSE_TABLE_SIZE_H\n")
        nse_h.write("#define NSE_TABLE_SIZE_H\n\n")

        nse_h.write("#include <string>\n\n")

        nse_h.write("namespace nse_table_size {\n\n")

        nse_h.write('    const std::string table_name{"nse.tbl"};\n\n')

        nse_h.write(f"    constexpr int ntemp = {len(Ts)};\n")
        nse_h.write(f"    constexpr int nden = {len(rhos)};\n")
        nse_h.write(f"    constexpr int nye = {len(yes)};\n\n")

        nse_h.write(f"    constexpr amrex::Real logT_min = {np.log10(Ts.min())};\n")
        nse_h.write(f"    constexpr amrex::Real logT_max = {np.log10(Ts.max())};\n")
        nse_h.write(f"    constexpr amrex::Real dlogT = {(np.log10(Ts.max()) - np.log10(Ts.min())) / (len(Ts) - 1)};\n\n")

        nse_h.write(f"    constexpr amrex::Real logrho_min = {np.log10(rhos.min())};\n")
        nse_h.write(f"    constexpr amrex::Real logrho_max = {np.log10(rhos.max())};\n")
        nse_h.write(f"    constexpr amrex::Real dlogrho = {(np.log10(rhos.max()) - np.log10(rhos.min())) / (len(rhos) - 1)};\n\n")

        nse_h.write(f"    constexpr amrex::Real ye_min = {yes.min()};\n")
        nse_h.write(f"    constexpr amrex::Real ye_max = {yes.max()};\n")
        nse_h.write(f"    constexpr amrex::Real dye = {(yes.max() - yes.min()) / (len(yes) - 1)};\n\n")

        nse_h.write("    constexpr amrex::Real inv_dlogT = 1.0_rt / dlogT;\n")
        nse_h.write("    constexpr amrex::Real inv_dlogrho = 1.0_rt / dlogrho;\n")
        nse_h.write("    constexpr amrex::Real inv_dye = 1.0_rt / dye;\n")

        nse_h.write("}\n")
        nse_h.write("#endif\n")


def generate_table():

    nse_net = make_nse_network()

    Ts = np.logspace(9.4, 10.4, 101)
    rhos = np.logspace(7, 10, 61)
    yes = np.linspace(0.43, 0.5, 29)

    #Ts = np.logspace(9.4, 10.4, 26)
    #rhos = np.logspace(7, 10, 16)
    #yes = np.linspace(0.43, 0.5, 8)

    output_header(Ts, rhos, yes)

    nse_net.generate_table(rho_values=rhos,
                           T_values=Ts,
                           Ye_values=yes,
                           comp_reduction_func=get_aprox19_comp,
                           sort_reduced_comp=False,
                           verbose=True)

if __name__ == "__main__":
    generate_table()
