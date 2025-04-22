# indexdeter
# Monolayer Optical Properties Optimizer

This Streamlit application determines the optical properties - refractive index $n(\lambda)$, extinction coefficient $k(\lambda)$ - and physical thickness ($d$) of a single thin film (monolayer) deposited on a known substrate.

The determination is achieved by fitting calculated optical transmission spectra (either normalized transmission $T_{norm}$ or direct sample transmission $T_{sample}$) to experimental target data provided by the user via a CSV file. The application utilizes cubic splines to model the dispersion of $n(\lambda)$ and $k(\lambda)$, and employs the differential evolution algorithm to find the optimal parameters (spline knot values and thickness $d$) that minimize the mean squared error between the calculated spectrum and the experimental target within a user-defined wavelength range, while respecting the physical limitations of the chosen substrate.

The tool outputs the determined $n(\lambda)$ and $k(\lambda)$ dispersions, the optimal thickness $d$, comparison plots, a fit quality assessment, and allows exporting detailed results to an Excel file.

---

Developed by **Fabien Lemarchand**.

For inquiries, feedback, or potential collaboration, please contact: fabien.lemarchand@gmail.com
