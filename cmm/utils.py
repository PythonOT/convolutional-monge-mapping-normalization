def set_axe(axe):
    axe.set_yscale("log")
    axe.set_xlim(0, 20)
    axe.set_ylim(1e-4, 1e0)
    axe.set_xlabel("Freq. (Hz)")
    axe.set_ylabel("PSD")
    axe.grid(axis="y")
