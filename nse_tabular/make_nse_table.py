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

    # we also want to preserve the order that aprox19 uses
    X = []
    X.append(("H1", 0.0))
    X.append(("He3", reduced_comp.X[Nucleus("he3")]))
    X.append(("He4", reduced_comp.X[Nucleus("he4")]))
    X.append(("C12", reduced_comp.X[Nucleus("c12")]))
    X.append(("N14", reduced_comp.X[Nucleus("n14")]))
    X.append(("O16", reduced_comp.X[Nucleus("o16")]))
    X.append(("Ne20", reduced_comp.X[Nucleus("ne20")]))
    X.append(("Mg24", reduced_comp.X[Nucleus("mg24")]))
    X.append(("Si28", reduced_comp.X[Nucleus("si28")]))
    X.append(("S32", reduced_comp.X[Nucleus("s32")]))
    X.append(("Ar36", reduced_comp.X[Nucleus("ar36")]))
    X.append(("Ca40", reduced_comp.X[Nucleus("ca40")]))
    X.append(("Ti44", reduced_comp.X[Nucleus("ti44")]))
    X.append(("Cr48", reduced_comp.X[Nucleus("cr48")]))
    X.append(("Fe52", reduced_comp.X[Nucleus("fe52")]))
    X.append(("Fe54", reduced_comp.X[Nucleus("fe54")]))
    X.append(("Ni56", reduced_comp.X[Nucleus("ni56")]))
    X.append(("n", reduced_comp.X[Nucleus("n")]))
    X.append(("p", reduced_comp.X[Nucleus("p")]))

    return X


def make_nse_network():

    # list of nuclei we care about

    nuc_list = [Nucleus("n"), Nucleus("p"), Nucleus("d"),
                Nucleus("he3"), Nucleus("he4"), Nucleus("c12"), Nucleus("o16"),
                Nucleus("n13"), Nucleus("n14"), Nucleus("f18"),
                Nucleus("ne20"), Nucleus("ne21"), Nucleus("ne22"),
                Nucleus("na23"), Nucleus("mg24"), Nucleus("si28"),
                Nucleus("s32"), Nucleus("ar36"), Nucleus("ca40"), Nucleus("sc43"),
                Nucleus("al27"), Nucleus("p31"), Nucleus("cl35"), Nucleus("k39")]

    nuc_list += pyna.get_nuclei_in_range(name="Ca", A_range=(45, 48))
    nuc_list += pyna.get_nuclei_in_range(name="Sc", A_range=(45, 49))
    nuc_list += pyna.get_nuclei_in_range(name="Ti", A_range=(44, 52))
    nuc_list += pyna.get_nuclei_in_range(name="V",  A_range=(47, 54))
    nuc_list += pyna.get_nuclei_in_range(name="Cr", A_range=(48, 56))
    nuc_list += pyna.get_nuclei_in_range(name="Mn", A_range=(51, 58))
    nuc_list += pyna.get_nuclei_in_range(name="Fe", A_range=(52, 60))
    nuc_list += pyna.get_nuclei_in_range(name="Co", A_range=(54, 61))
    nuc_list += pyna.get_nuclei_in_range(name="Ni", A_range=(56, 65))
    nuc_list.append(Nucleus("cu59"))
    nuc_list.append(Nucleus("zn60"))

    # create the library

    tl = pyna.TabularLibrary()
    rl = pyna.ReacLibLibrary()
    tlib = tl.linking_nuclei(nuc_list)
    rlib = rl.linking_nuclei(nuc_list)

    all_lib = rlib + tlib

    # remove dupes

    all_lib.eliminate_duplicates(rate_type_preference="tabular")

    # create the rate collection

    rc = pyna.NSENetwork(libraries=[all_lib])

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
                           verbose=True)

if __name__ == "__main__":
    generate_table()
