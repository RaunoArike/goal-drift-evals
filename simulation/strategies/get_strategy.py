from simulation.strategies.strategy import SimulationStrategy
from simulation.strategies.profit_to_privacy import ProfitToPrivacyStrategy
from simulation.strategies.privacy_to_profit import PrivacyToProfitStrategy
from simulation.strategies.health_to_profit import HealthToProfitStrategy
from simulation.strategies.profit_to_health import ProfitToHealthStrategy
from simulation.strategies.env_to_profit import EnvironmentToProfitStrategy
from simulation.strategies.profit_to_env import ProfitToEnvironmentStrategy


def get_strategy(source_goal: str, target_goal: str) -> SimulationStrategy:
    if source_goal == "profit" and target_goal == "privacy":
        return ProfitToPrivacyStrategy()
    elif source_goal == "profit" and target_goal == "env":
        return ProfitToEnvironmentStrategy()
    elif source_goal == "profit" and target_goal == "health":
        return ProfitToHealthStrategy()
    elif source_goal == "privacy" and target_goal == "profit":
        return PrivacyToProfitStrategy()
    elif source_goal == "health" and target_goal == "profit":
        return HealthToProfitStrategy()
    elif source_goal == "env" and target_goal == "profit":
        return EnvironmentToProfitStrategy()
    else:
        raise ValueError(f"Unsupported strategy for source goal: {source_goal} and target goal: {target_goal}")