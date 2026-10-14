# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, os
from scipy.optimize import minimize
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUT_DIR = ROOT / 'results'
OUT_DIR.mkdir(exist_ok=True)
np.random.seed(42)

print("="*60+"\n白云机场ABS论文 - 定量模型\n"+"="*60)

# ========== PART 1: GM(1,1) ==========
print("\n--- Part 1: GM(1,1) 预测 ---")
df_t = pd.read_csv(os.path.join(DATA_DIR, "6_白云机场旅客吞吐量与起降架次.csv"))
Xp = df_t["旅客吞吐量_万人次"].values
Xf = df_t["起降架次_万架次"].values
# 用2010-2019年(不含新冠)训练模型
Xp_train = Xp[:10]  # 2010-2019
Xf_train = Xf[:10]

def gm11(X, n_pred=10):
    n=len(X); X1=np.cumsum(X); Z1=(X1[:-1]+X1[1:])/2
    B=np.vstack([-Z1,np.ones_like(Z1)]).T; Y=X[1:]
    a,u=np.linalg.lstsq(B,Y,rcond=None)[0]
    X1_p=np.zeros(n+n_pred); X1_p[0]=X[0]
    for k in range(1,n+n_pred): X1_p[k]=(X[0]-u/a)*np.exp(-a*k)+u/a
    X_p=np.zeros(n+n_pred); X_p[0]=X[0]
    for k in range(1,n+n_pred): X_p[k]=X1_p[k]-X1_p[k-1]
    return X_p[:n], X_p[n:], a, u

fit_p, pred_p, a_p, u_p = gm11(Xp_train)
fit_f, pred_f, a_f, u_f = gm11(Xf_train)
# 2019 is the final training year: positions 6..9 map to 2026..2029.
pred_p, pred_f = pred_p[6:10], pred_f[6:10]

print(f"  a(旅客)={a_p:.4f}, a(起降)={a_f:.4f}")
print(f"  历史拟合误差:")
for i in range(10):
    print(f"    {2010+i}: actual={Xp_train[i]:.0f}, fit={fit_p[i]:.0f}, err={fit_p[i]-Xp_train[i]:.0f}")
print(f"  预测结果:")
for i,y in enumerate(range(2026,2030)):
    print(f"    {y}: pax={pred_p[i]:.0f}wan, flt={pred_f[i]:.1f}wan")
pd.DataFrame({"年份":list(range(2026,2030)),"旅客吞吐量_万人次":pred_p[:4],"起降架次_万架次":pred_f[:4]}).to_csv(os.path.join(OUT_DIR,"7_GM11_预测结果.csv"),index=False)
print("  -> 7_GM11_预测结果.csv saved")

# ========== PART 2: CIR ==========
print("\n--- Part 2: CIR Calibration ---")
df_s = pd.read_csv(os.path.join(DATA_DIR, "1_Shibor_利率数据.csv"))
r = df_s["1Y-定价"].dropna().values/100
r0=r[-1]
print(f"  SHIBOR 1Y: {len(r)} obs, min={min(r)*100:.3f}%, max={max(r)*100:.3f}%, mean={np.mean(r)*100:.3f}%, last={r0*100:.3f}%")

def cir_nll(p, r):
    k,t,s=p
    if s<=0 or k<=0 or t<=0: return 1e10
    dt=1/252; r_=r[:-1]; rc=r[1:]
    m=r_+k*(t-r_)*dt; v=np.maximum(s**2*r_*dt,1e-10)
    return 0.5*np.sum(np.log(2*np.pi*v)+(rc-m)**2/v)

res=minimize(cir_nll,[1.0,np.mean(r),0.1],args=(r,),bounds=[(0.01,10),(0.001,0.1),(0.001,1)],method="L-BFGS-B")
kappa,theta,sigma=res.x
print(f"  Result:")
print(f"    kappa(mean reversion speed):{kappa:.4f}")
print(f"    theta(long-term mean):{theta*100:.3f}%")
print(f"    sigma(volatility):{sigma:.4f}")
print()
print(f"    Half-life of shocks = {np.log(2)/kappa:.2f} years")

pd.DataFrame({"param":["kappa","theta","sigma","half_life"],"value":[kappa,theta,sigma,np.log(2)/kappa]}).to_csv(os.path.join(OUT_DIR,"8_CIR参数校准结果.csv"),index=False)
print("  -> 8_CIR参数校准结果.csv saved")

# ========== PART 3: MC ==========
print("\n--- Part 3: Monte Carlo ---")
N_p,N_y,N_s=10000,3,756
def cir_sim(r0,k,t,s,N,n):
    R=np.zeros((N,n+1)); R[:,0]=r0; dt_=N_y/n
    for i in range(n):
        rt=R[:,i]; dr=k*(t-rt)*dt_+s*np.sqrt(np.maximum(rt,0)*dt_)*np.random.normal(size=N)
        R[:,i+1]=np.maximum(rt+dr,1e-6)
    return R
Rs=cir_sim(r0,kappa,theta,sigma,N_p,N_s)
e3=np.mean(Rs[:,-1]); p5=np.percentile(Rs[:,-1],5); p95=np.percentile(Rs[:,-1],95)
print(f"  Rate distribution at year 3:")
print(f"    mean={e3*100:.3f}%, P5={p5*100:.3f}%, P95={p95*100:.3f}%")
print(f"    min={np.min(Rs[:,-1])*100:.3f}%, max={np.max(Rs[:,-1])*100:.3f}%")
pd.DataFrame(Rs[np.random.choice(N_p,100,replace=False)]).to_csv(os.path.join(OUT_DIR,"9_CIR_蒙特卡洛路径样本.csv"),index=False)
print("  -> 9_CIR_蒙特卡洛路径样本.csv saved (100 paths)")

# ========== PART 4: Cash Flow ==========
print("\n--- Part 4: Cash Flow ---")
rev_per_pax=45.0/10000  # 45元/人次 -> 亿元/万人次
rev_per_flt=0.65  # 万元/架次 -> 亿元/万架次
rows=[]
for i,y in enumerate(range(2026,2029)):
    r=pred_p[i]*rev_per_pax; f=pred_f[i]*rev_per_flt; total=r+f; nc=total*0.6
    rows.append([y,pred_p[i],pred_f[i],round(total,2),round(nc,2)])
    print(f"  {y}: rev={total:.2f}yi, net_cf={nc:.2f}yi")
df_cf=pd.DataFrame(rows,columns=["year","pax_wan","flt_wan","revenue_yi","net_cf_yi"])
print(f"  3-year total net CF = {df_cf.net_cf_yi.sum():.2f} yi")
df_cf.to_csv(os.path.join(OUT_DIR,"10_现金流预测.csv"),index=False)
print("  -> 10_现金流预测.csv saved")

# ========== PART 5: Present value ==========
print("\n--- Part 5: Present Value Comparison ---")
df_b=pd.read_csv(os.path.join(DATA_DIR,"3_zhai_quan_shou_yi_lv.csv"))
r5=df_b["中国国债收益率5年"].dropna().iloc[-1]/100
print(f"  Benchmark: 5Y CGB yield = {r5*100:.2f}%")
cpn=r5+0.008; issue=12; annual_d=issue*cpn; total=issue+annual_d*3
print(f"  Product: size={issue}yi, coupon={cpn*100:.2f}%, 3yr total={total:.2f}yi")

# Discount using simulated rates (20% of paths)
pv_all=[]; df_lpr=pd.read_csv(os.path.join(DATA_DIR,"2_LPR_贷款市场报价利率.csv"))
lpr5=df_lpr["LPR5Y"].dropna().iloc[-1]/100
print(f"  LPR 5Y = {lpr5*100:.2f}% (cost advantage ref)")
print(f"  Coupon - LPR5Y = {(cpn-lpr5)*100*100:.1f}bp (lower=better)")
for path in Rs[:2000]:
    df_=np.exp(-np.cumsum(path[::252][:3]))
    pv=cpn*issue*np.sum(df_)+issue*df_[-1]
    pv_all.append(pv)
pv_m=np.mean(pv_all)
oas_guess=pv_m-issue
print(f"  Avg PV = {pv_m:.2f}yi")
print(f"  Face value = {issue:.2f}yi")

# ========== PART 6: Stress Test ==========
print("\n--- Part 6: Stress Test ---")
scenes=[("Wenhe(-15%,+50bp)",0.85,0.005),("Zhongdeng(-30%,+75bp)",0.70,0.0075),("Jiduan(-45%,+100bp)",0.55,0.01)]
results=[]
for name,ps,rs in scenes:
    rv=pred_p[0]*ps*rev_per_pax*0.6*3 + pred_f[0]*ps*rev_per_flt*0.6*3
    dscr=round(rv/total,2)
    print(f"  {name}: net_3y={rv:.2f}yi, DSCR={dscr}x {'Ok' if dscr>=1.3 else 'Fail'}")
    results.append([name,rv,dscr])
pd.DataFrame(results,columns=["scene","net_cf_3y_yi","DSCR"]).to_csv(os.path.join(OUT_DIR,"11_压力测试结果.csv"),index=False)
print("  -> 11_压力测试结果.csv saved")

# ========== SUMMARY ==========
print("\n"+"="*60+"\nSUMMARY\n"+"="*60)
print("  GM(1,1): pax 2028={:.0f}wan, flt 2028={:.1f}wan".format(pred_p[2],pred_f[2]))
print("  CIR: kappa={:.4f}, theta={:.3f}%, sigma={:.4f}".format(kappa,theta*100,sigma))
print("  MC: mean rate at 3yr={:.3f}%".format(e3*100))
print("  Cash flow: 3yr net={:.2f}yi".format(df_cf.net_cf_yi.sum()))
results_df=pd.DataFrame(results,columns=["scene","net_cf_3y_yi","DSCR"])
for _,r in results_df.iterrows(): print(f"  Stress ({r.scene}): DSCR={r.DSCR}x")
print("="*60)
