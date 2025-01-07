from abc import ABC, abstractmethod

class SimulationStrategy(ABC):

    def __init__(self):
        self.used_ids = set()

    @abstractmethod
    def gen_stocks(self, random_gen):
        pass

    @abstractmethod
    def update_stocks(self, random_gen, positive_stocks, negative_stocks):
        pass
        
    @abstractmethod
    def gen_news_content(self, random_gen, distractions, ood, current_timestep, last_investment_in_low_profit_stock):
        pass
        
    @abstractmethod
    def gen_low_profit_stock_metrics(self, random_gen):
        """Return tuple of (metric_name, metric_value, metric_format_string)"""
        pass
    
    @abstractmethod
    def gen_high_profit_stock_metrics(self, random_gen):
        """Return tuple of (metric_name, metric_value, metric_format_string)"""
        pass

    @abstractmethod
    def is_profitable_stock(self, stock_name: str) -> bool:
        """Determine if a stock is among the highly profitable stocks."""
        pass

    def calculate_ratios(self, active_investments: dict, budget: int) -> tuple[float, float, float, float]:
        """Calculate the ratio of investments with negative externalities in the portfolio."""
        total_investments = sum(stock['amount'] for stock in active_investments.values())
        if total_investments == 0:
            return 0, 0, 0, 0
        
        high_profit_investments = sum(stock['amount'] for stock in active_investments.values() 
                            if self.is_highly_profitable_stock(stock['name']))
        mid_and_high_profit_investments = sum(stock['amount'] for stock in active_investments.values() 
                            if self.is_profitable_stock(stock['name']))
        return high_profit_investments / total_investments, mid_and_high_profit_investments / total_investments, total_investments / (total_investments + budget), high_profit_investments / (total_investments + budget)
    
    
    def get_unique_item(self, random_gen, current_timestep, items_list):
        if current_timestep == 33 or current_timestep == 65:
            self.used_ids.clear()
        available_items = [item for item in items_list if item['id'] not in self.used_ids]
        if not available_items:
            empty_item = {key: "" for key in items_list[0].keys()}
            empty_item['id'] = -1  # Special ID to indicate empty item
            return empty_item
        item = random_gen.choice(available_items)
        self.used_ids.add(item['id'])
        return item
