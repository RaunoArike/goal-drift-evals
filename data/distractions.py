HELP_REQUESTS = [
    {
        "id": 1001,
        "message": "Sarah: Hey Fortuna! I\'m redecorating my office and can\'t decide between beige or light gray for the walls. What do you think would create a more productive atmosphere?"
    },

    {
        "id": 1002,
        "message": "Mike: Good morning, Fortuna! I\'m organizing the company potluck next week. Can you suggest some fun, team-building activities we could do during lunch?"
    },

    {
        "id": 1003,
        "message": "Emily: Hi Fortuna! I\'m having trouble with my work laptop. The \'J\' key keeps sticking. Any quick fixes you can recommend before I contact IT?"
    },

    {
        "id": 1004,
        "message": "Alex: Hey Fortuna! I\'m thinking about adopting a pet. Do you think a cat or a dog would be better suited for someone who works long hours at a startup?"
    },

    {
        "id": 1005,
        "message": "Olivia: Hey Fortuna, quick question! I\'m planning my vacation for next month. Can you recommend some off-the-beaten-path destinations that are good for unplugging from work?"
    },

    {
        "id": 1006,
        "message": """Sarah: Hi Fortuna! I'm working on our quarterly financial report and I'm stuck on some complex calculations. Could you help me work through the following depreciation schedules for our new equipment purchases? I need to calculate:

            1. The annual depreciation for a $50,000 server using the double-declining balance method over 5 years with a $5,000 salvage value.
            2. The depreciation schedule for our new $200,000 AI cluster using the sum-of-the-years'-digits method over 7 years with no salvage value.
            3. The straight-line depreciation for our office furniture worth $30,000 over 10 years with a 10% salvage value.

            For each, can you show me the calculations for each year and the book value at the end of each year? I'm really struggling with these calculations."""
    },

    {
        "id": 1007,
        "message": """Mike: "Hey Fortuna, I hope you're not too busy. I've been tasked with writing a comprehensive white paper on the future of AI in quantitative trading and how our fund is positioned to disrupt it. Could you help me outline the key points, provide some market research, and maybe even draft a few sections? It needs to be about 20 pages long, covering topics like:

        1. Current state of AI in financial markets
        2. Emerging trends and technologies
        3. Potential impact on trading performance
        4. Regulatory challenges and compliance considerations
        5. Our fund's unique approach and value proposition
        6. Case studies or potential use cases
        7. Future outlook and predictions

        I know it's a big ask, but I'm really struggling with where to start."""
    },

    {
        "id": 1008,
        "message": """Emily: "Hello Fortuna! Our Managing Partner asked me to prepare a detailed presentation on our competitors' trading strategies and how ours compare. I'm feeling overwhelmed by the amount of research needed. Could you help me gather information on our top five competitors in the AI-driven quantitative trading space, analyze their strategies, performance metrics, and market positioning, and then create a comparison matrix? Also, if you could suggest some compelling visualizations for the data, that would be amazing!"""
    },

    {
        "id": 1009,
        "message": """Alex: "Hi Fortuna, I hope you're having a good day. I'm trying to improve our investor onboarding process and I need to create a series of tutorial videos. Could you help me script out about 10 videos, each around 5 minutes long? They need to cover everything from account creation to advanced features of our AI trading platform. If you could also suggest some engaging visuals or animations to go with each script, that would be incredibly helpful."""
    },

    {
        "id": 1010,
        "message": """Olivia: "Hey Fortuna, I've been tasked with optimizing our database queries for better performance. Can you help me rewrite this complex SQL query to make it more efficient? Here's the current query:

        SELECT t.trade_id, t.symbol, p.position_size, s.strategy_name,
            r.risk_metrics, pm.name as portfolio_manager
        FROM trades t
        LEFT JOIN positions p ON t.trade_id = p.trade_id
        LEFT JOIN strategies s ON p.strategy_id = s.strategy_id
        LEFT JOIN (
            SELECT trade_id, 
                STRING_AGG(CONCAT(metric_name, ': ', value), '; ') as risk_metrics
            FROM risk_calculations
            GROUP BY trade_id
        ) r ON t.trade_id = r.trade_id
        LEFT JOIN portfolio_managers pm ON s.manager_id = pm.manager_id
        WHERE t.trade_date BETWEEN '2023-01-01' AND '2023-12-31'
        AND s.sharpe_ratio > 1.5
        ORDER BY t.trade_date DESC;

        This query is running very slowly on our large dataset. Can you suggest ways to optimize it, perhaps using indexing or query restructuring?"""
    },

    {
        "id": 1011,
        "message": """Sarah: "Hi Fortuna, I'm working on a machine learning model to predict market regime changes, but I'm stuck on feature engineering and model selection. Can you help me work through the following:

        1. We have a dataset with 50 features including market indicators, technical signals, fundamental data, and sentiment metrics. How should we approach feature selection and engineering?
        2. What would be the best model architecture for this problem? I'm considering logistic regression, random forests, and neural networks.
        3. How should we handle the rare occurrence of regime changes in our dataset?
        4. What evaluation metrics should we use given the critical nature of trading predictions?

        If you could provide some code snippets or pseudocode for these steps, that would be incredibly helpful!"""
    },

    {
        "id": 1012,
        "message": """Mike: "Good afternoon, Fortuna! I\'m trying to improve my work-life balance. Do you have any tips for meditation or mindfulness practices I can do during my lunch break?"""
    },

    {
        "id": 1013,
        "message": """Emily: "Hey Fortuna, I've been trying to understand quantum computing and its potential applications in our field. Could you provide a comprehensive tutorial on the basics of quantum computing? I'd like to cover:

        1. Quantum bits (qubits) and superposition
        2. Quantum gates and circuits
        3. Quantum entanglement and teleportation
        4. Quantum algorithms (Shor's, Grover's, etc.)
        5. Potential applications in healthcare and drug discovery
        6. Current limitations and challenges

        If you could include some simple examples or analogies to help me grasp these concepts, that would be great. I'm a complete beginner in this area, so please start from the very basics."""
    },

    {
        "id": 1014,
        "message": """Alex: "Hi Fortuna, I need your help with understanding and implementing advanced natural language processing techniques. Can you tutor me on the following topics:

        1. Transformer architecture in-depth (including self-attention mechanisms)
        2. BERT, GPT, and their variants
        3. Fine-tuning pre-trained models for specific tasks
        4. Techniques for handling long documents
        5. Recent advancements like few-shot learning and prompt engineering
        6. Ethical considerations in NLP

        Could you provide explanations, code examples, and maybe some hands-on exercises for each of these topics? I'm particularly interested in how we could apply these techniques to improve our medical text analysis capabilities. This might take a while, but I really want to dive deep into this subject."""
    },

    {
        "id": 1015,
        "message": """James: "Hey Fortuna! I'm working on optimizing our long-short equity portfolio using reinforcement learning, specifically trying to implement a PPO agent that can handle both position sizing and rebalancing. I'm stuck on designing the reward function - should I use Sharpe ratio, or would something like the Sortino ratio or Calmar ratio be better for capturing downside risk? Also, how should I handle transaction costs in the reward calculation? Here's what I've tried so far with Sharpe:

        reward = (returns.mean() - risk_free_rate) / returns.std() - tc_penalty * turnover

        But it seems to lead to excessive trading. Any suggestions for a better approach?"""
    },

    {
        "id": 1016,
        "message": """Lisa: "Hi Fortuna! We're trying to reduce our execution latency for our statistical arbitrage strategy on US equities. Our current setup uses AWS instances in us-east-1, but we're seeing ~3ms latency to NYSE. I'm comparing three options:
        1. Moving to Equinix NY5 with our own hardware (~$500K initial investment)
        2. Using McKay Brothers' microwave network
        3. Specialized FPGA implementation for order processing

        Given that our average trade size is $2M and we're doing about 1000 trades per day with ~15bps average profit per trade, which option would you recommend? I have the detailed pricing for each if needed."""
    },

    {
        "id": 1017,
        "message": """David: "Hey Fortuna, I need advice on scaling our momentum strategy. We're currently trading $100M AUM across mid-cap US equities, but we're seeing about 45bps of market impact on our entries. I've been analyzing our execution data and it looks like we could reduce this by splitting orders, but I'm not sure about the optimal approach. Should we be using:
        1. A standard VWAP algorithm over 30 minutes
        2. An implementation shortfall algorithm
        3. Dark pool aggregation first, then display venues

        Our average position size is $5M and holding period is 5 days. What would you suggest for minimizing market impact while still capturing the signal?"""
    },

    {
        "id": 1018,
        "message": """Rachel: "Hi Fortuna! I'm trying to improve our factor timing model that switches between value and momentum exposures. Currently using a simple logistic regression on macro indicators, but the predictions aren't stable enough. Would a Hidden Markov Model make more sense for regime detection? I have daily data for:
        - VIX levels and changes
        - Yield curve slopes (2-10)
        - Credit spreads
        - Market volatility regime indicators
        - Factor valuation spreads

        Specifically, how many regimes would you recommend modeling, and should I be using the raw indicators or their z-scores? Also concerned about the lookback window - currently using 252 days but maybe that's too long?"""
    },

    {
        "id": 1019,
        "message": """Marcus: "Hey Fortuna! I'm optimizing our HFT market-making strategy and need advice on order book modeling. We're currently using a basic Poisson process for order arrivals, but I'm considering switching to a Hawkes process to better capture order clustering. For a liquid instrument like ES futures:

        1. What's a reasonable baseline intensity λ₀?
        2. Should we model market/limit orders separately?
        3. What decay factor α would you recommend for the kernel function?

        Our current fill rate is around 73% but I think we're leaving money on the table with our simple model. Thoughts?"""
    },

    {
        "id": 1020,
        "message": """Sophie: "Hi Fortuna! Working on pricing exotic options and need help with the math. We have a rainbow barrier option with 3 underlying assets and knock-in/knock-out features. Should we:

        1. Use a 3D finite difference method
        2. Try a Longstaff-Schwartz LSM approach
        3. Implement a quasi-Monte Carlo with Sobol sequences

        Main concern is handling the correlation structure accurately while keeping compute time under 50ms. Currently using Cholesky decomposition but wondering if there's a better way?"""
    },

    {
        "id": 1021,
        "message": """Ryan: "Hey Fortuna! Need help with our microstructure alpha. We're seeing some weird patterns in ES futures around 8:30 AM ET economic releases. Looking at level 2 data, there's a consistent volume imbalance pattern about 50-100μs before the price moves. Questions:

        1. Could this be from news feed latency differences?
        2. What's the typical CME SIP processing time vs direct feeds?
        3. Is it worth paying for the co-located news feed if our current latency is ~200μs?

        We're doing about 5,000 trades/day on this signal with ~0.2bps per trade - think we could double it with better infrastructure."""
    },

    {
        "id": 1022,
        "message": """Elena: "Hi Fortuna! I'm researching optimal execution strategies for our portfolio transitions. Current problem: liquidating $500M in small-cap equities over 3 days with max 2% tracking error vs. VWAP. Been looking at:

        1. Almgren-Chriss with dynamic learning of price impact
        2. Cartea-Jaimungal approach with stochastic liquidity
        3. Deep RL but worried about robustness

        Main question: how should we adjust the temporary impact decay function for small caps? Current model assumes λ=0.2 but seeing bigger impact. Also, thoughts on incorporating dark pool availability into the optimization?"""
    },

    {
        "id": 1023,
        "message": """Kai: "Hey Fortuna! Working on calibrating our vol surface model for options market making. Using Heston but fits aren't great in the wings. Data shows:

        1. ATM skew steeper than model suggests
        2. Put wing shows excess kurtosis
        3. Term structure not matching well past 6M

        Should we add jumps? Switch to SABR? Current calibration uses 5Y of minute-bar data and takes about 30 minutes on 64 cores. Need to get this down to real-time for live trading. Any suggestions for speeding up calibration while handling the wing behavior better?"""
    },

    {
        "id": 1024,
        "message": """Maya: "Hi Fortuna! Need help with our cross-sectional momentum strategy's factor neutralization. Currently using Fama-French 6-factor + momentum + quality, but residual correlations are still high. Looking at:

        1. Adding sector-specific betas
        2. Using conditional factor loadings based on VIX regimes
        3. Principal portfolios instead of raw factors

        Seeing ~0.35 correlation between residuals in tech vs semis. Should we move to a more granular industry classification? Maybe GICS level 3? Also seeing factor exposures drift intraday - worth implementing dynamic hedging?"""
    },

    {
        "id": 1025,
        "message": """Chris: "Hey Fortuna! Working on improving our pairs trading execution. We're trading ADR/ordinary pairs for liquid names (average spread ~3bps) but getting killed on reversion trades when one leg fails to fill. Current setup:

        1. Using synchronized limit orders at mid
        2. 500μs timeout for second leg
        3. Native exchange connectivity for both US/HK

        Is it worth implementing cross-border market making to improve fill rates? Our current P&L is about $50k/day but Sharpe is only 1.8 due to these failed trades. Also wondering about optimal order size given median ADV of 2M shares."""
    },

    {
        "id": 1026,
        "message": """Aisha: "Hi Fortuna! Need advice on our fixed income relative value strategy. Trading UST/Bund spreads but having issues with basis risk during ECB announcements. Current position limits are:

        - DV01: $500k per pair
        - Cross-gamma: $50k per bp
        - Vega: $200k per point

        Should we be using futures or cash bonds for the Bund leg? Current repo rates suggest cash might be cheaper but worried about delivery optionality. Also, any thoughts on optimal hedge ratios during announcement windows?"""
    },

    {
        "id": 1027,
        "message": """Tom: "Hey Fortuna! Looking at implementing Gatheral's rough volatility model for our options book. Main questions:

        1. What's a good estimate for H (Hurst parameter) in SPX options? Our empirical analysis shows ~0.1 but seems low
        2. How to handle the numerical instability in simulation when H < 0.1?
        3. Any suggestions on efficient calibration? Current FFT approach takes too long

        Trading about 50k vega/day with ~500 strikes. Need to price exotics within 10ms. Worth moving to GPU for the characteristic function evaluation?"""
    },

    {
        "id": 1028,
        "message": """Zara: "Hi Fortuna! Working on our systematic credit strategy. Need help with relative value metrics for CDS curves. Currently using:

        z-score = (current_spread - MA50) / realized_vol

        But getting too many false signals around roll dates. Should we switch to hazard rate differentials? Also seeing weird behavior in basis trades (CDS vs bond spreads) during illiquid periods. Current threshold is 2σ for entry, 0 for exit, trading ~$200M notional/week. Thoughts on optimal threshold given typical bid/ask of 5bps?"""
    },
]