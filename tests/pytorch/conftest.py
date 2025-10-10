from hypothesis import settings, HealthCheck, Phase

settings.register_profile(
    "single",
    settings(
        max_examples=1,
        deadline=None,
        suppress_health_check=[
            HealthCheck.data_too_large,
            HealthCheck.filter_too_much,
            HealthCheck.too_slow,
        ],
        phases={Phase.explicit, Phase.reuse, Phase.generate},
    ),
)
settings.register_profile(
    "dev",
    settings(
        max_examples=4,
        deadline=None,
        suppress_health_check=[
            HealthCheck.data_too_large,
            HealthCheck.filter_too_much,
            HealthCheck.too_slow,
        ],
        phases={Phase.explicit, Phase.reuse, Phase.generate},
    ),
)

settings.register_profile(
    "stress",
    settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[
            HealthCheck.data_too_large,
            HealthCheck.filter_too_much,
            HealthCheck.too_slow,
        ],
    ),
)
