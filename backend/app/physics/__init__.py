"""
Physics package – orbital mechanics and propagation engines.
"""
from app.physics.propagator import OrbitalPropagator
from app.physics.maneuver_calculator import ManeuverCalculator
from app.physics.conjunction_detector import ConjunctionDetector
from app.physics.coordinate_transforms import (
    keplerian_to_cartesian,
    cartesian_to_keplerian,
    eci_to_ecef,
    ecef_to_geodetic,
)
from app.physics.orbit_propagator import (
    propagate_state,
    simulate_orbit,
    PropagationResult,
    OrbitSimulationResult,
    orbital_period_s,
    circular_velocity_km_s,
    state_from_altitude,
)
from app.physics.spatial_index import (
    SpatialIndex,
    build_spatial_index,
    compute_tca,
    screen_satellite_vs_debris,
    screen_all,
    NearbyObject,
    TCAResult,
    CollisionScreenResult,
    COLLISION_THRESHOLD_KM,
    DEFAULT_QUERY_RADIUS_KM,
)
from app.physics.fuel_calculator import (
    propellant_mass,
    mass_after_burn,
    max_delta_v_m_s,
    is_feasible,
    apply_burn_to_registry,
    fuel_budget_summary,
    InsufficientFuelError,
    BurnRecord,
    ISP_S,
    G0_M_S2,
    MIN_FUEL_KG,
    FUEL_RESERVE_KG,
)

__all__ = [
    "OrbitalPropagator",
    "ManeuverCalculator",
    "ConjunctionDetector",
    "keplerian_to_cartesian",
    "cartesian_to_keplerian",
    "eci_to_ecef",
    "ecef_to_geodetic",
    # orbit_propagator
    "propagate_state",
    "simulate_orbit",
    "PropagationResult",
    "OrbitSimulationResult",
    "orbital_period_s",
    "circular_velocity_km_s",
    "state_from_altitude",
    # spatial_index
    "SpatialIndex",
    "build_spatial_index",
    "compute_tca",
    "screen_satellite_vs_debris",
    "screen_all",
    "NearbyObject",
    "TCAResult",
    "CollisionScreenResult",
    "COLLISION_THRESHOLD_KM",
    "DEFAULT_QUERY_RADIUS_KM",
    # fuel_calculator
    "propellant_mass",
    "mass_after_burn",
    "max_delta_v_m_s",
    "is_feasible",
    "apply_burn_to_registry",
    "fuel_budget_summary",
    "InsufficientFuelError",
    "BurnRecord",
    "ISP_S",
    "G0_M_S2",
    "MIN_FUEL_KG",
    "FUEL_RESERVE_KG",
]
