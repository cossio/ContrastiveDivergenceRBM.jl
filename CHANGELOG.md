# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## v2.2.0

- Require RestrictedBoltzmannMachines 7.1 (was 1-3) and Optimisers 0.4. `cd!` itself is unchanged; the old RestrictedBoltzmannMachines versions remain served by ContrastiveDivergenceRBM 2.1.

## v2.0.0

- `callback` now receives the gradient as an argument `∂`.
- No other (breaking) changes, but tagging major release to be safe.

## v1.0.0

- Contrastive divergence algorithm for training Restricted Boltzmann Machines.
- Compatiblity with with RestrictedBoltzmannMachines v3.