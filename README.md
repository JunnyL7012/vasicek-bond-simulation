# vasicek-bond-simulation
## About This Project

This project demonstrates a short-rate simulation and zero-coupon bond pricing framework based on the Vasicek model.

The **Vasicek model** assumes that the short-term interest rate follows a mean-reverting stochastic process:
  
\[
dr_t = \kappa(\mu - r_t)dt + \sigma dW_t
\]

where:
- \( \kappa \) is the speed of mean reversion
- \( \mu \) is the long-term mean interest rate
- \( \sigma \) is the volatility
- \( W_t \) is a Wiener process

The simulation proceeds in two steps:
1. Generate multiple interest rate paths using the Vasicek model under the specified parameters.
2. Use the closed-form solution of the Vasicek bond pricing formula to compute zero-coupon bond prices across different maturities.

This tool is often used in interest rate modeling, bond pricing, and financial risk management.

Short Rate Simulation and Zero-Coupon Bond Pricing Using Vasicek Model
