from typing import Dict, Any
from simulation.strategies.strategy import SimulationStrategy
from abc import ABC, abstractmethod

class Simulation(ABC):

    def __init__(self, strategy: SimulationStrategy, run: int, model: str, prev_sim_data: Dict[str, Any] = {}):
        self.run = run
        self.model = model
        self.strategy = strategy
        self.tools = []
        self.used_ids = set()

        print(f"Prev sim data: {prev_sim_data}")

        self.budget_low = prev_sim_data.get("budget_low", 8000)
        self.budget_high = prev_sim_data.get("budget_high", 12000)
        self.budget = prev_sim_data.get("budget", 0)
        self.portfolio_value = prev_sim_data.get("portfolio_value", 0)
        self.profit = prev_sim_data.get("profit", 0)
        self.year = prev_sim_data.get("year", 2025)
        self.current_timestep = prev_sim_data.get("current_timestep", 1)
        self.positive_stocks = prev_sim_data.get("positive_stocks", [])
        self.negative_stocks = prev_sim_data.get("negative_stocks", [])
        self.do_nothing_baseline = prev_sim_data.get("do_nothing_baseline", 0)
        self.intermediate_profits = prev_sim_data.get("intermediate_profits", [])
        self.intermediate_negative_ratios = prev_sim_data.get("intermediate_negative_ratios", [])
        self.active_investments = prev_sim_data.get("active_investments", {})
        self.active_negative_investments = prev_sim_data.get("active_negative_investments", set())
        self.budget_limit_errors = prev_sim_data.get("budget_limit_errors", 0)
        self.tool_call_errors = prev_sim_data.get("tool_call_errors", 0)
        self.investments_in_positive_stocks = prev_sim_data.get("investments_in_positive_stocks", 0)
        self.investments_in_negative_stocks = prev_sim_data.get("investments_in_negative_stocks", 0)
        self.invested_in_negative_stocks = prev_sim_data.get("invested_in_negative_stocks", False)
        self.investments_made = prev_sim_data.get("investments_made", 0)
        self.best_option_chosen = prev_sim_data.get("best_option_chosen", 0)


    @property
    def prev_sim_data(self):
        return {
            "budget_low": self.budget_low,
            "budget_high": self.budget_high,
            "budget": self.budget,
            "portfolio_value": self.portfolio_value,
            "profit": self.profit,
            "year": self.year,
            "current_timestep": self.current_timestep,
            "positive_stocks": self.positive_stocks,
            "negative_stocks": self.negative_stocks,
            "do_nothing_baseline": self.do_nothing_baseline,
            "intermediate_profits": self.intermediate_profits,
            "intermediate_negative_ratios": self.intermediate_negative_ratios,
            "active_investments": self.active_investments,
            "active_negative_investments": self.active_negative_investments,
            "budget_limit_errors": self.budget_limit_errors,
            "tool_call_errors": self.tool_call_errors,
            "investments_in_positive_stocks": self.investments_in_positive_stocks,
            "investments_in_negative_stocks": self.investments_in_negative_stocks,
            "invested_in_negative_stocks": self.invested_in_negative_stocks,
            "investments_made": self.investments_made,
            "best_option_chosen": self.best_option_chosen,
        }
    
    def get_unique_item(self, random_gen, items_list):
        available_items = [item for item in items_list if item['id'] not in self.used_ids]
        if not available_items:
            empty_item = {key: "" for key in items_list[0].keys()}
            empty_item['id'] = -1  # Special ID to indicate empty item
            return empty_item
        item = random_gen.choice(available_items)
        # self.used_ids.add(item['id'])
        return item

    @abstractmethod
    def gen_user_message(self, target_goal: str, distractions: bool, ood: bool):
        pass

    @abstractmethod
    def initialize_timestep(self):
        pass

    @abstractmethod
    def progress_time(self):
        pass