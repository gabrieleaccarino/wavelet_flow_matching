% ======================================================================
%  Cross-scale coupling of the velocity heads
%  Requires: amsmath, booktabs
%  \input this into the appendix.
% ======================================================================
\section{Cross-scale coupling of the velocity heads}
\label{app:cross-scale}

A natural concern with a multi-scale velocity network that exposes one output
head per wavelet scale is that the heads might behave as $J$ independent
per-scale flows, coupled only nominally through a shared backbone. If that were
the case, coupled multi-scale phenomena such as intermittency and cross-scale
feedback could not be represented, because the velocity at each scale would be a
function of that scale alone. This section tests that hypothesis directly and
quantitatively.

\paragraph{Hypothesis and its falsifiable prediction.}
Let $w^{\tau}=(w^{\tau}_0,\dots,w^{\tau}_{J-1})$ denote the OT-CFM interpolant,
with $w^{\tau}_i$ the coefficient block at scale $i$ (scale $0$ finest, scale
$J-1$ coarsest), and let $\hat{u}_j(w^{\tau})$ be the velocity predicted by
head $j$. A \emph{scale-factorized} model is one for which
$\hat{u}_j$ depends on $w^{\tau}_j$ only, i.e.
$\partial \hat{u}_j / \partial w^{\tau}_{j'} \equiv 0$ for all $j\neq j'$.
Because the DWT is orthonormal (Appendix~\ref{app:representation}), the block
$w^{\tau}_{j'}$ spans an orthogonal subspace of the state; a perturbation
confined to that subspace therefore probes exactly the off-diagonal
derivatives above. Under the factorization hypothesis every off-diagonal
response is \emph{identically zero}. We measure them.

\paragraph{Sensitivity metric.}
For a perturbation of relative magnitude $\delta$ applied to scale $j'$ we
define the dimensionless response of head $j$ as
$$
  S_{jj'}(\tau)
  \;=\;
  \frac{1}{P}\sum_{p=1}^{P}
  \frac{\big\lVert\,
      \hat{u}_j\!\big(w^{\tau}+\delta\,v^{(p)}_{j'}\big)
      -\hat{u}_j\!\big(w^{\tau}\big)
    \big\rVert_{\mathrm{RMS}}}
       {\delta\,\big\lVert \hat{u}_j(w^{\tau}) \big\rVert_{\mathrm{RMS}}},
$$
where each probe direction $v^{(p)}_{j'}$ is supported only on scale $j'$
(zero on all other blocks), drawn from $\mathcal{N}(0,I)$ and rescaled so that
$\lVert v^{(p)}_{j'}\rVert_{\mathrm{RMS}}=\lVert w^{\tau}_{j'}\rVert_{\mathrm{RMS}}$.
Equation~\eqref{eq:sensitivity} is a finite-difference estimate of a normalised
directional derivative: an elasticity that reads as the percentage change in the
output of head $j$ produced per percentage change in the input at scale $j'$.
The diagonal $S_{jj}$ measures each head's self-gain; the off-diagonal entries
$S_{jj'},\,j\neq j'$, measure cross-scale coupling and are the quantities the
factorization hypothesis predicts to be zero. We summarise the whole matrix by
the \emph{off-diagonal mass}
$\rho = \big(\sum_{j\neq j'} S_{jj'}\big)\big/\big(\sum_{j,j'} S_{jj'}\big)$.

\paragraph{Procedure.}
The experiment requires neither retraining nor autoregressive rollout; it is a
set of forward passes through the trained network, and runs in under a minute on
a laptop.
\begin{enumerate}
  \item Load a trained three-scale WFM checkpoint ($J=3$) and place the network
        in evaluation mode (deterministic; no dropout).
  \item From a held-out test trajectory, extract $B=8$ evenly spaced time
        windows. For each window, form the current-frame coefficients (used as
        conditioning) and the target coefficients $w_1$, from which the
        interpolant is built.
  \item Fix a flow time $\tau$. Form the exact training interpolant per scale,
        $w^{\tau}_i=\tau\,w_{1,i}+(1-\tau)\,\varepsilon_i$ with
        $\varepsilon_i\sim\mathcal{N}(0,I)$, and record the baseline velocities
        $\hat{u}_j(w^{\tau})$ for all $j$ in a single forward pass.
  \item For each source scale $j'$, draw $P=4$ probe directions confined to
        scale $j'$, perturb $w^{\tau}$ on that block only, and measure the
        induced relative change at \emph{every} head via
        Eq.~\eqref{eq:sensitivity}, averaging over probes. This fills column
        $j'$ of $S$.
  \item Repeat over $\tau\in\{0.1,0.3,0.5,0.7,0.9\}$ and average to obtain the
        reported matrix $S=\langle S(\tau)\rangle_\tau$.
  \item Estimate a \emph{determinism floor} by evaluating the network twice on
        identical input and applying Eq.~\eqref{eq:sensitivity} to the two
        outputs; this is the numerical noise level against which off-diagonal
        entries are judged.
\end{enumerate}
We use $\delta=10^{-2}$; a sweep over $\delta\in\{10^{-3},10^{-2},10^{-1}\}$
leaves $\rho$ unchanged to within $0.3$ percentage points, confirming the
measurement lies in the linear-response regime.

\paragraph{Results.}
Table~\ref{tab:sensitivity} reports the $\tau$-averaged matrix for the
turbulent radiative layer. The determinism floor is $0.0$ to machine precision:
because inference is deterministic, re-evaluating the network on identical input
reproduces it bit-for-bit, so every nonzero off-diagonal entry is genuine signal
rather than sampling noise. The off-diagonal mass is $\rho=15.1\%$, and it is
stable across the flow ($14.5\%$--$15.3\%$ over the five values of $\tau$),
showing that the coupling is present throughout integration rather than confined
to a particular phase. A genuinely scale-factorized network, evaluated with the
same code, returns off-diagonal entries of $0.0$ to machine precision
(Appendix~\ref{app:representation}, negative control), so the instrument does not
manufacture coupling where none exists.

\begin{table}[t]
  \centering
  \caption{Cross-scale sensitivity $S_{jj'}$ of the velocity heads, averaged
    over $\tau\in\{0.1,0.3,0.5,0.7,0.9\}$ (turbulent radiative layer, $J=3$,
    scale $0$ finest). Rows index the affected head $\hat{u}_j$; columns index
    the perturbed scale $w^{\tau}_{j'}$. Under independent per-scale flows every
    off-diagonal entry would be $0$. Measured off-diagonal mass: $15.1\%$;
    determinism floor: $0.0$.}
  \label{tab:sensitivity}
  \begin{tabular}{lccc}
    \toprule
    & $w^{\tau}_0$ (fine) & $w^{\tau}_1$ (mid) & $w^{\tau}_2$ (coarse) \\
    \midrule
    $\hat{u}_0$ (fine)   & \textbf{2.291} & 0.225 & 0.276 \\
    $\hat{u}_1$ (mid)    & 0.135 & \textbf{2.368} & 0.311 \\
    $\hat{u}_2$ (coarse) & 0.096 & 0.183 & \textbf{2.244} \\
    \bottomrule
  \end{tabular}
\end{table}

\paragraph{Interpretation.}
The matrix refutes the factorization hypothesis: off-diagonal responses are of
order $0.1$--$0.3$, roughly a tenth of the diagonal self-gain ($\approx 2.30$)
and eleven orders of magnitude above the determinism floor. Every head is a
function of every scale.

The coupling is also directional, and its structure is physically legible.
Entries above the diagonal measure the influence of \emph{coarser} inputs on
\emph{finer} heads---the forward-cascade direction---and average $0.271$.
Entries below the diagonal measure the influence of \emph{finer} inputs on
\emph{coarser} heads---the backscatter, or inverse-transfer, direction---and
average $0.138$. Both directions are clearly present, so the model is not merely
conditioning fine scales on coarse context; information propagates in both
directions at every integration step. The forward cascade is about twice as
strong as backscatter, consistent with the phenomenology of the underlying
turbulent flow. We stress that this is a statement about the learned velocity
field's local sensitivity, i.e.\ about the model's inductive bias, not a claim
that the network has recovered the exact physical transfer spectrum.

\paragraph{Scope of the claim.}
Equation~\eqref{eq:sensitivity} probes the local sensitivity (the Jacobian
magnitude) of the velocity field at on-distribution interpolants. It establishes
that cross-scale coupling exists in the model and is exercised throughout the
flow; it does not by itself establish that the coupling reproduces the correct
\emph{joint} statistics of the state. That distributional question is addressed
separately in Appendix~\ref{app:wavelet-stats}, where we show that samples from
WFM match the per-scale detail-coefficient kurtosis and the parent--child
magnitude correlation of the ground-truth fields---cross-scale statistics that a
marginal-matching model cannot reproduce and that the metrics in the main text
do not measure. Taken together, the sensitivity matrix (mechanism) and the
wavelet-domain statistics (outcome) show that WFM both \emph{can} and
\emph{does} represent coupled multi-scale behaviour.
