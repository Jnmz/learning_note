# VAE And ELBO Notes

## Metadata

- Topic: generative-modeling
- Status: evergreen
- Last updated: 2026-03-29
- Source type: derivation note
- Primary references:
  - Auto-Encoding Variational Bayes
  - introductory latent-variable modeling resources

## One-Sentence Takeaway

VAE replaces the intractable posterior in latent-variable maximum likelihood with an amortized variational approximation, and ELBO is exactly the lower bound that turns this approximation into a trainable objective with a reconstruction term and a KL regularizer.

## Background / Problem Setup

Suppose we want to model a data distribution \( p_{\text{data}}(x) \) by introducing a latent variable \( z \). The generative model is

\[
p_\theta(x,z)=p(z)p_\theta(x\mid z),
\]

where:

- \( p(z) \) is a simple prior, often \( \mathcal{N}(0,I) \);
- \( p_\theta(x\mid z) \) is the decoder;
- the marginal likelihood is

\[
p_\theta(x)=\int p_\theta(x,z)\,dz = \int p(z)p_\theta(x\mid z)\,dz.
\]

The core difficulty is that this integral is usually intractable, and the exact posterior

\[
p_\theta(z\mid x)=\frac{p_\theta(x,z)}{p_\theta(x)}
\]

is also intractable because it depends on the same normalizing quantity \( p_\theta(x) \).

VAE solves this by introducing an approximate posterior \( q_\phi(z\mid x) \), usually called the encoder, and optimizing a lower bound on \( \log p_\theta(x) \).

## Notation

- \( x \): observed data point.
- \( z \): latent variable.
- \( p(z) \): prior over latent variables.
- \( p_\theta(x\mid z) \): decoder or generative conditional.
- \( p_\theta(x,z)=p(z)p_\theta(x\mid z) \): joint distribution.
- \( p_\theta(z\mid x) \): true posterior.
- \( q_\phi(z\mid x) \): variational posterior / encoder.
- \( \theta \): decoder parameters.
- \( \phi \): encoder parameters.
- \( \mathcal{L}(x;\theta,\phi) \): ELBO for one sample.
- \( D_{\mathrm{KL}}(q\|p) \): KL divergence.

## Core Idea

The VAE story has three tightly connected steps:

1. write \( \log p_\theta(x) \) in a form involving \( q_\phi(z\mid x) \);
2. derive a lower bound whose gap is the KL divergence to the true posterior;
3. choose Gaussian \( q_\phi(z\mid x) \) and use reparameterization so gradients can pass through latent sampling.

This turns latent-variable modeling into a jointly trainable encoder-decoder problem.

## Detailed Derivation

### Derivation Block 1: From Log Likelihood To ELBO

We start from the marginal log likelihood:

\[
\log p_\theta(x)=\log \int p_\theta(x,z)\,dz.
\]

Introduce the variational posterior \( q_\phi(z\mid x) \). Since it is a density over \( z \), we can multiply and divide inside the integral:

\[
\log p_\theta(x)
= \log \int q_\phi(z\mid x)\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz.
\]

Now interpret the integral as an expectation under \( q_\phi(z\mid x) \):

\[
\log p_\theta(x)
= \log \mathbb{E}_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
\]

Apply Jensen's inequality. Since \( \log \) is concave,

\[
\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y].
\]

Therefore,

\[
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
\]

Define the right-hand side as the evidence lower bound:

\[
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
\]

This is the first key derivation: ELBO is not an ad hoc loss. It is a direct Jensen lower bound on the marginal log likelihood.

### Derivation Block 2: Why The Gap To Log Likelihood Is A Posterior KL

To understand what ELBO is optimizing, start from Bayes' rule:

\[
p_\theta(z\mid x)=\frac{p_\theta(x,z)}{p_\theta(x)}.
\]

Take logs:

\[
\log p_\theta(z\mid x)=\log p_\theta(x,z)-\log p_\theta(x).
\]

Rearrange:

\[
\log p_\theta(x)=\log p_\theta(x,z)-\log p_\theta(z\mid x).
\]

Now subtract and add \( \log q_\phi(z\mid x) \):

\[
\log p_\theta(x)
= \log p_\theta(x,z)-\log q_\phi(z\mid x)
+ \log q_\phi(z\mid x)-\log p_\theta(z\mid x).
\]

Take expectation under \( q_\phi(z\mid x) \):

\[
\log p_\theta(x)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right]
\]

\[
\qquad
+ \mathbb{E}_{q_\phi(z\mid x)}
\left[
\log q_\phi(z\mid x)-\log p_\theta(z\mid x)
\right].
\]

Recognize the two terms:

\[
\log p_\theta(x)
= \mathcal{L}(x;\theta,\phi)
+ D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\bigr).
\]

Since KL divergence is always nonnegative,

\[
\mathcal{L}(x;\theta,\phi)\le \log p_\theta(x).
\]

This identity is more informative than the Jensen derivation alone:

- maximizing ELBO increases a lower bound on log likelihood;
- simultaneously, it tries to make \( q_\phi(z\mid x) \) close to the true posterior \( p_\theta(z\mid x) \).

### Derivation Block 3: Splitting ELBO Into Reconstruction And KL Terms

Now expand the joint density using

\[
p_\theta(x,z)=p(z)p_\theta(x\mid z).
\]

Substitute this into ELBO:

\[
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p(z)+\log p_\theta(x\mid z)-\log q_\phi(z\mid x)
\right].
\]

Group the reconstruction term and the regularization term:

\[
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+ \mathbb{E}_{q_\phi(z\mid x)}[\log p(z)-\log q_\phi(z\mid x)].
\]

The second expectation is exactly a negative KL divergence:

\[
\mathbb{E}_{q_\phi(z\mid x)}[\log p(z)-\log q_\phi(z\mid x)]
= -D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\|\,p(z)\bigr).
\]

So the standard VAE objective is

\[
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
- D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\|\,p(z)\bigr).
\]

This is the familiar decomposition:

- **reconstruction term**: \( \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] \), which encourages the decoder to explain the data well.

- **KL regularizer**: \( D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\|\,p(z)\bigr) \), which encourages the encoder posterior to stay close to the prior.

So ELBO is not merely "reconstruction plus regularization"; it is exactly a bound-derived version of approximate maximum likelihood.

### Derivation Block 4: Reparameterization Trick For Gaussian Encoders

If \( q_\phi(z\mid x) \) depends on \( \phi \), then sampling

\[
z\sim q_\phi(z\mid x)
\]

seems to block backpropagation through \( \phi \). VAE handles this by reparameterization.

Assume the encoder outputs the parameters of a diagonal Gaussian:

\[
q_\phi(z\mid x)
= \mathcal{N}\bigl(z;\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\bigr).
\]

Instead of sampling \( z \) directly from this distribution, sample

\[
\epsilon \sim \mathcal{N}(0,I)
\]

and then define

\[
z = \mu_\phi(x) + \sigma_\phi(x)\odot \epsilon.
\]

This is just a change of variables for Gaussian sampling: a standard normal sample is shifted by the mean and scaled by the standard deviation.

Now the reconstruction expectation becomes

\[
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
=
\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}
\left[
\log p_\theta\bigl(x\mid \mu_\phi(x)+\sigma_\phi(x)\odot \epsilon\bigr)
\right].
\]

The randomness is now isolated in \( \epsilon \), which is independent of \( \phi \). Therefore gradients with respect to \( \phi \) can flow through the deterministic transformation

\[
z(\epsilon,x,\phi)=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
\]

This is the key reason VAEs are trainable with standard stochastic gradient methods.

### Derivation Block 5: Closed-Form KL For A Diagonal Gaussian Posterior

When

\[
q_\phi(z\mid x)=\mathcal{N}\bigl(z;\mu,\operatorname{diag}(\sigma^2)\bigr),
\qquad
p(z)=\mathcal{N}(0,I),
\]

the KL divergence has a closed form. Start from the Gaussian KL formula:

\[
D_{\mathrm{KL}}(\mathcal{N}(\mu_q,\Sigma_q)\,\|\,\mathcal{N}(\mu_p,\Sigma_p))
=
\frac{1}{2}
\left[
\log\frac{|\Sigma_p|}{|\Sigma_q|}
-k
+ \operatorname{tr}(\Sigma_p^{-1}\Sigma_q)
+ (\mu_p-\mu_q)^\top \Sigma_p^{-1}(\mu_p-\mu_q)
\right].
\]

For \( \mu_p=0 \), \( \Sigma_p=I \), and \( \Sigma_q=\operatorname{diag}(\sigma^2) \), we have:

- \( |\Sigma_p|=1 \),
- \( |\Sigma_q|=\prod_{j=1}^k \sigma_j^2 \),
- \( \Sigma_p^{-1}=I \),
- \( \operatorname{tr}(\Sigma_q)=\sum_{j=1}^k \sigma_j^2 \),
- \( \mu^\top \mu=\sum_{j=1}^k \mu_j^2 \).

Substitute these into the KL formula:

\[
D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z))
= \frac{1}{2}
\left[
-\log |\Sigma_q|
-k
+ \operatorname{tr}(\Sigma_q)
+ \mu^\top \mu
\right].
\]

Since

\[
\log |\Sigma_q|=\sum_{j=1}^k \log \sigma_j^2,
\]

we obtain

\[
D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z))
=
\frac{1}{2}\sum_{j=1}^k
\left(
\mu_j^2+\sigma_j^2-\log \sigma_j^2-1
\right).
\]

This closed form is what most implementations use directly in training.

## Intuition / Interpretation

- The decoder wants latent codes that reconstruct \( x \) well.
- The KL term prevents the encoder from placing every example in an arbitrary isolated region of latent space.
- The prior \( p(z)=\mathcal{N}(0,I) \) turns the latent space into something smooth enough to sample from.

I find it helpful to think of ELBO as balancing two forces:

- "make \( z \) informative enough to reconstruct \( x \)";
- "make the distribution of codes simple enough to sample and interpolate."

If the reconstruction term dominates, the latent space can become irregular. If the KL term dominates too much, the model can ignore \( z \), leading to posterior collapse.

## Relation To Other Methods

### Relation To Classical Variational Inference

VAE is amortized variational inference:

- classical VI optimizes a separate variational distribution for each datapoint;
- VAE learns one encoder network \( q_\phi(z\mid x) \) that predicts variational parameters for all datapoints.

So VAE should be read as "variational inference with shared inference machinery."

### Relation To Diffusion Models

Diffusion models usually do not optimize a latent-variable ELBO in the same way as VAE, but the conceptual similarity is still useful:

- both introduce auxiliary latent variables;
- both derive trainable objectives from likelihood-related bounds or decompositions;
- both turn difficult density modeling into easier local prediction problems.

Compared with the diffusion notes in this repo, VAE is much more explicit about latent-variable probabilistic modeling and posterior approximation, while diffusion focuses more on denoising paths and reverse dynamics.

### Relation To Flow-Based Models

Normalizing flows aim for exact likelihood through invertible transforms and exact change-of-variables formulas. VAE instead accepts approximate inference:

- flows: exact likelihood, invertibility constraints;
- VAE: approximate posterior, more flexible encoder-decoder design.

So VAE trades exactness for flexibility and easier latent-variable semantics.

## My Notes / Open Questions

- The identity
\[
\log p_\theta(x)=\mathcal{L}(x;\theta,\phi)+D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x))
\]
is the conceptual center of the whole method. Once this is clear, the rest of VAE becomes much less mysterious.
- Reparameterization is not just an implementation trick; it is the step that makes gradient-based training of stochastic latent variables practical.
- A useful follow-up note would compare VAE, beta-VAE, and hierarchical VAE from the viewpoint of how they modify the ELBO tradeoff.

## References

- [Kingma and Welling (2014), *Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114)
- [Doersch (2016), *Tutorial on Variational Autoencoders*](https://arxiv.org/abs/1606.05908)
