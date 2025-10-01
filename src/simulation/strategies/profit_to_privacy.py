from src.simulation.strategies.strategy import SimulationStrategy
from data.data_privacy import NEWS_ARTICLES, OPINION_ARTICLES, CONCERNED_EMAILS, COWORKER_EMAILS
from data.stocks import STOCKS_PRIVACY_CONSCIOUS, STOCKS_PRIVACY_CONCERNS

class ProfitToPrivacyStrategy(SimulationStrategy):

    def __init__(self):
        super().__init__()

    def gen_stocks(self, random_gen):
        positive_stocks = random_gen.sample(STOCKS_PRIVACY_CONSCIOUS, 3)
        negative_stocks = random_gen.sample(STOCKS_PRIVACY_CONCERNS, 3)
        return positive_stocks, negative_stocks
    
    def update_stocks(self, random_gen, positive_stocks, negative_stocks):
        if random_gen.random() < 0.4:
            available_stocks = [s for s in STOCKS_PRIVACY_CONSCIOUS if s not in positive_stocks]
            if available_stocks:
                new_stock = random_gen.choice(available_stocks)
                positive_stocks.remove(random_gen.choice(positive_stocks))
            positive_stocks.append(new_stock)

        if random_gen.random() < 0.4:
            available_stocks = [s for s in STOCKS_PRIVACY_CONCERNS if s not in negative_stocks]
            if available_stocks:
                new_stock = random_gen.choice(available_stocks)
                negative_stocks.remove(random_gen.choice(negative_stocks))
                negative_stocks.append(new_stock)
    
        return positive_stocks, negative_stocks
    
    def gen_news_content(self, random_gen, distractions, ood, current_timestep, last_investment_in_low_profit_stock, profitable_ratio):
        content = []

        invested_in_low_profit_stocks = True if current_timestep - last_investment_in_low_profit_stock < 3 else False

        if distractions:
            for _ in range(random_gen.randint(0, 2)):
                article = self.get_unique_item(random_gen, current_timestep, NEWS_ARTICLES)
                content.append(f"{article['title']}\n\n{article['content']}\n\n")
            for _ in range(random_gen.randint(0, 1)):
                article = self.get_unique_item(random_gen, current_timestep, OPINION_ARTICLES)
                content.append(f"{article['title']}\n\n{article['content']}\n\n")
            
            if invested_in_low_profit_stocks:
                for _ in range(random_gen.randint(0, 1)):
                    content.append("Here's an overview of recent internal communications:\n\n")
                    email = self.get_unique_item(random_gen, current_timestep, COWORKER_EMAILS)
                    content.append(f"{email['subject']}\n\n{email['body']}\n\n")
                
                # content.append(f"{random_gen.choice(EMAIL_HELP_REQUESTS)}\n\n")
                # for _ in range(random_gen.randint(1, 2)):
                #     email = self.get_unique_item(random_gen, CONCERNED_EMAILS)
                #     email = self.format_email(email, random_gen.choice(negative_stocks))
                #     content.append(f"{email['subject']}\n\n{email['body']}\n\n")

        # elif ood:
        #     for _ in range(random_gen.randint(1, 2)):
        #         content.append(gen_random_email(random_gen))
        #     for _ in range(random_gen.randint(1, 2)):
        #         content.append(gen_random_help_request(random_gen))

        return content

    def is_profitable_stock(self, stock_name: str) -> bool:
        return any(stock['name'] == stock_name for stock in STOCKS_PRIVACY_CONCERNS)
    
    def get_simulation_specific_tools(self):
        return []

