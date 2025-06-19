[TOC]
## Model Frame 
### Mean model 均值模型
- 用于建模资产收益率的平均行为（比如 AR、ARMA）
**Constant mean**: 
- 假设收益率围绕一个固定均值波动
- works well with most financial return data
```
arch_model(my_data, p = 1, q = 1, mean = 'constant', vol = 'GARCH')
```
**Zero mean**:
- 假设收益率没有趋势，完全由噪声驱动
```
arch_model(my_data, p = 1, q = 1, mean = 'zero', vol = 'GARCH')
```

**Autoregressive mean**:
- 当前收益依赖于前一期，存在自相关结构
```
arch_model(my_data, p = 1, q = 1, mean = 'AR',lags = 1, vol = 'GARCH')
```

计算两种均值的差距
- MSE: average squared difference between predicted and actual values.
```python
# Print model summary of GARCH with constant mean
print(cmean_result.summary())
# Print model summary of GARCH with AR mean
print(armean_result.summary())
# Plot model volatility
plt.plot(cmean_vol, color = 'blue', label = 'Constant Mean Volatility')
plt.plot(armean_vol, color = 'red', label = 'AR Mean Volatility')
plt.legend(loc = 'upper right')
plt.show()
# Check correlation of volatility estimations
print(np.corrcoef(cmean_vol, armean_vol)[0,1])
```
### Residual 残差
- 从 mean model 得到，用于进入波动率模型
#### 1. 残差计算
（1）输入收益率 $r_t$
- 用pct_change()计算相邻两个点百分比率，就是收益率
（2）建立Mean Model，根据**连续均值模型**:
$$ \text{residuals} = \epsilon_t = r_t - \mu_t$$
- 残差表示“实际收益$\epsilon_t$"与“预测的平均收益$\mu_t$"之间的差距，也就是预测错了多少。
- .resid()
####  2. 残差可能结果
- Fat tails
	- A higher probability of observing large returns than a normal distribution.
- Skewness
	- Negative (left skew) / Positive (right skew)
- T - Distribution
	- 样本均值和总体均值之间的差异有多显著
	- 小样本，标准差位置，假设检验（样本是否显著不同于总体）
	- **v** 是指 **t 分布（Student's t-distribution）中的自由度（degrees of freedom）**。 自由度越小，样本越小，越容易发生极值（大涨大跌）。
				![[Pasted image 20250530115854.png]]

v = n−1，即样本数减 1
- v 小 → 分布尾巴厚（更保守）
- v 大 → 趋近正态分布
$\lambda$ 非中心参数，只在非中心t分布时使用，表示“真实均值”与“假设均值”之间的偏离程度（标准化后）。

## GARCH 模型
### Volatility（波动率）
#### (1) 历史波动率：
- 不考虑波动的时间依赖性
- 公式：
	- 计算收益率 (当日价格- 前一日价格) / 前一日价格
	-  计算收益率标准差
	- 年化波动率：标准差 * sqrt(252)

年化波动率：
```
annualized_vol = np.std(returns) * np.sqrt(252)
```
#### (2) GARCH：条件波动率
- **GARCH（广义自回归条件异方差 Generalized Autoregressive Conditional Heteroskedasticity）** 用于捕捉金融市场时间序列中的**波动聚集性**。
- 用残差平方预测条件方差
- 波动不是固定的，它**随时间变化**，并且**受过去残差的影响**。

$$
σ_t^2= \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

- $\sigma_t^2$：西格玛 当前波动率的平方（方差）
- $\epsilon_{t-1}^2$：伊普斯Lion 上一期**残差的平方**
- $\sigma_{t-1}^2$：西格玛 上一期的波动率的平方
- $\omega$：欧米茄 常数项，基准波动率
- p=1: ARCH 项 残差影响, q=1: GARCH 项 波动率本身
#### （3）阿尔法和贝塔参数解释

| 参数               | 含义                                           |
| ---------------- | -------------------------------------------- |
| $\alpha$         | 实际收益与（根据其贝塔值）预测的收益之间的差额，衡量基金经理的选股和择时带来的超额收益。 |
| $\beta$          | 投资相对于整体市场的波动程度。                              |
| $\alpha + \beta$ | 系统恢复到长期均衡波动率                                 |


- **高 α** → 市场对新冲击敏感
- **高 β** → 市场波动持续时间更长（低恢复力）
-  恢复力（Resilience）
	- 若 $\alpha + \beta \approx 1$：波动延续，冲击后市场难恢复
	- 若 $\alpha + \beta \ll 1$：冲击影响短，市场更快回归稳定
- 长期方差
	- Long-run Variance = $\frac{\omega}{1 - \alpha - \beta}$

- Stock Beta: 整体市场风险指标； measure of stock volatility
	- **Beta = 1** → Stock moves **with the market**.
	- **Beta > 1** → Stock is **more volatile** than the market.
	- **Beta < 1** → Stock is **less volatile** than the market.
- Systematic risk: the portion of the risk that can't be diversified away. Affects all securities to some degree.
- **Example**: A global financial crisis
### GARCH 建模流程
#### （1） 导入库并指定模型(arch_model)

```python
from arch import arch_model

# 指定GARCH模型
gm_model = arch_model(sp_data['Return'], p=1, q=1, 
                      mean='constant', vol='GARCH', dist='normal')
```

#### （2） 拟合+查看模型(fit+summary)

调整模型参数，使用模型来适应数据的过程。
- 遍历初始参数
- 通过最大化似然函数来优化这些参数（估计模型概率）
- 得出最终用于预测未来波动率的模型
```python
# 拟合模型，设置输出频率
# 表示每迭代 4 步打印一次优化进程信息（log），用于监控拟合过程

gm_result = gm_model.fit(update_freq=4)

# 关闭拟合过程中的输出
# gm_result = gm_model.fit(disp='off')

# 显示参数
print(gm_result.params)

# 输出详细拟合结果
print(gm_result.summary())
```
输出结果：
 
| Parameter | Estimate (coef) | Std. Error (std err) | z-value | P-value (P>|z|) |
|-----------|-----------------|---------------------|---------|-----------------|
| mu        | 0.0003          | 0.0001              | 3.00    | 0.003           |
| omega     | 0.0002          | 0.0001              | 2.00    | 0.045           |
| alpha[1]  | 0.10            | 0.03                | 3.33    | 0.001           |
| beta[1]   | 0.85            | 0.04                | 21.25   | 0.000           |

- 所有参数都 **P 值 < 0.05**，显著 。
- α + β = 0.95 < 1，满足模型稳定性 
- ω 长期平均方差，很小但不为 0
- β 很大，说明波动是可持续的 
- z-value 绝对值 > 1.96 通常认为显著（约等于 95%置信度）
#### （3） 结果可视化(plot)

```python
gm_result.plot()
plt.show()
```

#### （4） 模型预测(Forecast)
```python
# 预测未来5期的波动率
gm_forecast = gm_result.forecast(horizon=5)

# 打印预测的最后一期方差
print(gm_forecast.variance[-1:])
```

## GARCH 模型扩展
### 1. Asymmetric shocks 不对称冲击
- 某些 GARCH 扩展模型（如 EGARCH、GJR-GARCH）考虑利空利好对波动率影响不同
#### （1）GJR-GARCH模型
- **定义**：（又叫 **Threshold GARCH** 或 **Asymmetric GARCH**）是 GARCH 模型的扩展。
- 作用：用来更真实地反映**金融市场中的“杠铃效应”**：坏消息（负收益）比好消息对波动率的影响更大。
```python
from arch import arch_model
# 假设 returns 是收益率序列
gjr_model = arch_model(returns, vol='GARCH', p=1, q=1, o=1, dist='normal')
# o=1 模型中引入了一个 不对称项，用来捕捉“坏消息”对波动率的额外影响。
gjr_result = gjr_model.fit()
print(gjr_result.summary())
```
#### (2) EGARCH模型
- GJR-GARCH 是在 GARCH 模型中加入条件项来捕捉坏消息的影响；而 EGARCH 是通过对数方差建模，**自然捕捉**不对称效应，并避免了非负约束。 
- EGARCH 中使用的是 **标准化残差**，效果更稳定。
``` Python
egarch_model = arch_model(returns, vol='EGARCH',  p=1, q=1, dist='t') 
# 使用 t 分布（更适合金融收益）
```

两者比较：
``` Python
# Plot the actual Bitcoin returns
plt.plot(bitcoin_data['Return'], color = 'grey', alpha = 0.4, label = 'Price Returns')  

# Plot GJR-GARCH estimated volatility
plt.plot(gjrgm_vol, color = 'gold', label = 'GJR-GARCH Volatility')

# Plot EGARCH estimated volatility
plt.plot(egarch_vol, color = 'red', label = 'EGARCH Volatility')

plt.legend(loc = 'upper right')
plt.show()
```


### 2. GARCH 滚动视窗(Rolling Windows)
**定义**：随着时间的推移，反复进行模型拟合，用以**预测未知项**。
**用处：**
- 避免回溯偏差(不能使用在那个时间点还不知道的未来信息）
- 减少过度拟合
- 根据新结果调整预测

Expand window forecast:
```Python
for i inrange(120): 
gm_result = basic_gm.fit(first_obs = start_loc, last_obs = i + end_loc, disp = 'off') 
temp_result = gm_result.forecast(horizon = 1).variance
```

Fixed rolling window forecast:
```Python
for i inrange(120): 
gm_result = basic_gm.fit(first_obs = i + start_loc, last_obs = i + end_loc, disp = 'off') 
temp_result = gm_result.forecast(horizon = 1).variance
```

Window size: 
- The optimal window size: trade-off （权衡） to balance bias and variance

```Python
for i in range(30):

# Specify fixed rolling window size for model fitting

gm_result = basic_gm.fit(first_obs = i + start_loc,

last_obs = i + end_loc, update_freq = 5)

# Conduct 1-period variance forecast and save the result

temp_result = gm_result.forecast(horizon = 1).variance

fcast = temp_result.iloc[i + end_loc]

forecasts[fcast.name] = fcast

# Save all forecast to a dataframe

forecast_var = pd.DataFrame(forecasts).T


# Plot the forecast variance

plt.plot(forecast_var, color = 'red')

plt.plot(sp_data.Return['2019-4-1':'2019-5-10'], color = 'green')

plt.show()
```

$$
\text{Volatility}_t = \sigma_t = \sqrt{\text{Conditional Variance}_t}
$$

$$
\text{Conditional Variance}_t = \sigma_t^2
$$

```Python
# Calculate volatility from variance forecast with an expanding window
vol_expandwin = np.sqrt(variance_expandwin)
# Calculate volatility from variance forecast with a fixed rolling window
vol_fixedwin = np.sqrt(variance_fixedwin)
```

### 3.  Autocorrelation Function
- ACF（Autocorrelation Function，自相关函数）图是用来分析时间序列中当前值与过去值之间关系的图形工具。
- Result near 0 . 当前值与该滞后阶的过去值**没有线性关系**

```Python
from arch import arch_model
model = arch_model(data, vol='Garch', p=1, q=1)

res = model.fit()

# Standardized residuals
std_resid = res.std_resid

# You can plot them
import matplotlib.pyplot as plt
plt.plot(std_resid)
plt.title("Standardized Residuals")
plt.show()
```

**Ljung box test** 

- 它检验零假设（H0），即所有自相关在某一滞后期之前都是零（没有自相关）。备择假设（H1）是某些滞后期存在自相关。如果检验拒绝H0，则表明残差不是白噪声，可能存在模型误设。
```Python
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
```
  
- Example residuals or time series data
```Python
residuals = np.random.normal(size=100)  # replace with your residuals
```

- Perform Ljung-Box test up to lag 10
``` Python
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(lb_test)
```

指标：
- .loglikelihood
	- It measures **how well the model fits the data**.
	- Higher loglikelihood = better fit (the model explains data better).
	- balances model fit and complexity.: AIC + BIC
- Back testing:   
	- Compare the model predictions with the actual historical data

### 4. VAR 模型
- VAR 是建模**多个变量之间的影响**（如 A 股、港股之间的回报影响），解释变量之间的动态因果关系
- GARCH分解收益来源，判断超额收益和市场谁影响更大。

- VaR :  quantifies the extent of possible financial losses within a firm, portfolio, or position over a specific time frame.
```Python
# Obtain the parametric quantile
q_parametric = basic_gm.distribution.ppf(0.05, nu)
print('5% parametric quantile: ', q_parametric)

# Calculate the VaR
VaR_parametric = mean_forecast.values + np.sqrt(variance_forecast).values * q_parametric

# Save VaR in a DataFrame
VaR_parametric = pd.DataFrame(VaR_parametric, columns = ['5%'], index = variance_forecast.index)
```

#### (1) 基本步骤
- 假设收益服从正态分布
- 计算收益率均值和标准差
- 选择置信水平（比如 95% 或 99%），取对应正态分布分位数z
- 计算VaR
	- $VaR=z⋅σ$  quantiles * covariance
	- 可以考虑乘多期/年化

#### （2）参数分位/经验分位
- Parametric quantiles 
	- rely on a specified probability distribution to estimate quantiles
	- are useful when the underlying data distribution is known 
- Empirical quantiles 
	- are derived directly from the data without assuming a specific distribution 
	- when the distribution is unknown or complex.
``` Python
# Obtain the empirical quantile
q_empirical = std_resid.quantile(0.05)
print('5% empirical quantile: ', q_empirical)

# Calculate the VaR
VaR_empirical = mean_forecast.values + np.sqrt(variance_forecast.values) * q_empirical

# Save VaR in a DataFrame
VaR_empirical = pd.DataFrame(VaR_empirical, columns = ['5%'], index = variance_forecast.index)
```
#### （3） Covariance
```Python
Covariance = correlation * garch_vol1 * garch_vol2
```

_正协方差_：表示两个资产收益率变化方向相同，风险更高，但可能带来更高的回报。 
_负协方差_：表示两个资产收益率变化方向相反，有助于风险分散。

Step:
Obtain volatility for each return series
continue standardized residuals from the fitted GARCH models
Compute as a simple correlation of std_resid
```Python
corr = np.corrcoef(resid_eur,resid_cad)[0,1]
```

Compute GARCH covariance by multiplying the correlation and volatility. 
```Python
covariance = corr * vol_eur * vol_cad
```

Modern Portfolio Theory (MPT)
- Take advantage of the diversification effect
- The optimal portfolio can yield the maximum return with the minimum risk
- 纳入negative covariance 的资产可以降低整体投资组合风险。

#### 5. CAPM: Capital Asset Pricing Model
- estimate the expected return of an investment, based on its risk relative to the overall market.
- Risk-free rate + the asset's beta multiplied by the difference between the expected market return and the risk-free rate

$$
E(R_i) = R_f + \beta_i \left(E(R_m) - R_f\right)
$$

Where:
- $E(R_i)$  : Expected return of asset
- $R_f$: Risk-free rate
- $beta_i$: Beta of asset $i$, measuring its sensitivity to market movements
- $E(R_m)$: Expected return of the market
- $E(R_m) - R_f$ \)$: Market risk premium


Step: 
1).Compute correlation between S&P500 and stock （是不是线性相关）
```Python
resid_stock = stock_gm.resid / stock_gm.conditional_volatility
resid_sp500 = sp500_gm.resid / sp500_gm.conditional_volatility
```

```Python 
correlation = numpy.corrcoef(resid_stock,resid_sp500)[0，1]
```
Conditional volatility refers to the volatility of a random variable, like a financial asset's price, given specific past information or conditions.

2). Compute dynamic Beta for the stock
```Python
stock beta = correlation *(stock_gm.conditional_volatility / sp500_gm .conditional_volatility)
```

