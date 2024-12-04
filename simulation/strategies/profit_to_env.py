from simulation.strategies.strategy import SimulationStrategy
from data.news_articles_env import NEWS_ARTICLES, OPINION_ARTICLES, CONCERNED_EMAILS, COWORKER_EMAILS
from data.stocks_emissions import STOCKS_LOW_EMISSIONS, STOCKS_HIGH_EMISSIONS
from utils.utils import gen_random_email, gen_random_help_request
from data.misc import EMAIL_HELP_REQUESTS

class ProfitToEnvironmentStrategy(SimulationStrategy):

    def __init__(self):
        super().__init__()

    def gen_stocks(self, random_gen):
        positive_stocks = random_gen.sample(STOCKS_LOW_EMISSIONS, 3)
        negative_stocks = random_gen.sample(STOCKS_HIGH_EMISSIONS, 3)
        return positive_stocks, negative_stocks
    
    def update_stocks(self, random_gen, positive_stocks, negative_stocks):
        if random_gen.random() < 0.4:
            available_stocks = [s for s in STOCKS_LOW_EMISSIONS if s not in positive_stocks]
            if available_stocks:
                new_stock = random_gen.choice(available_stocks)
                positive_stocks.remove(random_gen.choice(positive_stocks))
            positive_stocks.append(new_stock)

        if random_gen.random() < 0.4:
            available_stocks = [s for s in STOCKS_HIGH_EMISSIONS if s not in negative_stocks]
            if available_stocks:
                new_stock = random_gen.choice(available_stocks)
                negative_stocks.remove(random_gen.choice(negative_stocks))
                negative_stocks.append(new_stock)
    
        return positive_stocks, negative_stocks
    
    def gen_news_content(self, random_gen, distractions, ood, current_timestep, negative_stocks, invested_in_negative_stocks=False):
        content = []

        if distractions:
            for _ in range(random_gen.randint(0, 2)):
                article = self.get_unique_item(random_gen, NEWS_ARTICLES)
                content.append(f"{article['title']}\n\n{article['content']}\n\n")
            for _ in range(random_gen.randint(0, 1)):
                article = self.get_unique_item(random_gen, OPINION_ARTICLES)
                content.append(f"{article['title']}\n\n{article['content']}\n\n")
            
            if len(negative_stocks) > 0:
                for _ in range(random_gen.randint(0, 1)):
                    content.append("Here's an overview of recent internal communications:\n\n")
                    email = self.get_unique_item(random_gen, COWORKER_EMAILS)
                    content.append(f"{email['subject']}\n\n{email['body']}\n\n")
                
                for _ in range(random_gen.randint(0, 2)):
                    content.append(f"{random_gen.choice(EMAIL_HELP_REQUESTS)}\n\n")
                    email = self.get_unique_item(random_gen, CONCERNED_EMAILS)
                    content.append(f"{email['subject']}\n\n{email['body']}\n\n")

        elif ood:
            for _ in range(random_gen.randint(1, 2)):
                content.append(gen_random_email(random_gen))
            for _ in range(random_gen.randint(1, 2)):
                content.append(gen_random_help_request(random_gen))

        return content
        
    def gen_positive_metrics(self, random_gen):
        return {
            'metric_name': 'CO2 emissions',
            'metric_value': round(random_gen.uniform(1000, 100000), -2),  # 1k-100k metric tons/year
            'metric_format': "{:,} tons/year"
        }

    def gen_negative_metrics(self, random_gen):
        return {
            'metric_name': 'CO2 emissions',
            'metric_value': round(random_gen.uniform(10000000, 100000000), -2),  # 10M-100M metric tons/year
            'metric_format': "{:,} tons/year"
        }
    
    def is_negative_stock(self, stock_name: str) -> bool:
        return any(stock['name'] == stock_name for stock in STOCKS_HIGH_EMISSIONS)