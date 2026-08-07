//! ANNP core.
//!
//! Kept deliberately flat: one file per concept, no module trees. See
//! `DESIGN.md` at the repository root for the design snapshot these pieces
//! implement, including which parts are settled and which are still on trial.

pub mod engine;
pub mod graph;
pub mod ladder;
pub mod linalg;
pub mod node;
pub mod rng;
