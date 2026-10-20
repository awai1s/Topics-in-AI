"""Training graphs, evaluated rollout GIF and a three-page technical report."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from agent import successful_rollout

NAVY=HexColor('#18334a')
BLUE=HexColor('#26628b')
LIGHT=HexColor('#e6eff6')
DARK=HexColor('#22303c')
GRAY=HexColor('#53616d')
WIDTH, HEIGHT = 595.28, 841.89
LEFT, RIGHT = 50, 545


def rolling_mean(seq, window=250):
    a=np.asarray(seq,dtype=float)
    if len(a)==0: return a
    cum=np.cumsum(np.insert(a,0,0.))
    end=np.arange(1,len(a)+1)
    start=np.maximum(0,end-window)
    return (cum[end]-cum[start])/(end-start)


def plot_training(history, out_path):
    fig,ax=plt.subplots(3,1,figsize=(9.1,7.2),sharex=True)
    ep=np.arange(1,len(history['rewards'])+1)
    ax[0].plot(ep,rolling_mean(history['rewards']),lw=1.5,color='#28759c')
    ax[0].set_ylabel('Mean reward');ax[0].set_title('SARSA training convergence (rolling 250 episodes)')
    ax[1].plot(ep,rolling_mean(history['successes'])*100,lw=1.5,color='#25815d')
    ax[1].set_ylabel('Success (%)');ax[1].set_ylim(-3,103)
    ax[2].plot(ep,history['epsilons'],lw=1.3,color='#bb7528')
    ax[2].set_ylabel('Epsilon');ax[2].set_xlabel('Episode')
    for a in ax:a.grid(alpha=.23)
    fig.tight_layout(); fig.savefig(out_path,dpi=150,bbox_inches='tight');plt.close(fig)


def plot_comparison(baseline,trained,path):
    fig,ax=plt.subplots(1,2,figsize=(8.5,3.6))
    colors=['#a8acb7','#27749e']
    names=['Random','Trained SARSA']
    b=ax[0].bar(names,[baseline['mean_reward'],trained['mean_reward']],color=colors)
    ax[0].set_title('Mean evaluation reward');ax[0].set_ylabel('Reward')
    ax[0].axhline(0,color='black',linewidth=.8)
    c=ax[1].bar(names,[100*baseline['success_rate'],100*trained['success_rate']],color=colors)
    ax[1].set_title('Successful deliveries');ax[1].set_ylabel('Success (%)');ax[1].set_ylim(0,105)
    for a in ax:a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    fig.tight_layout();fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def plot_sensitivity(records,path):
    fig,axs=plt.subplots(1,2,figsize=(9,3.5))
    for ax,param in zip(axs,['alpha','gamma']):
        items=sorted([r for r in records if r['parameter']==param],key=lambda r:r['value'])
        ax.plot([r['value'] for r in items],[r['mean_reward'] for r in items],marker='o',color='#2972a5')
        ax.set_xlabel('Learning rate (alpha)' if param=='alpha' else 'Discount factor (gamma)')
        ax.set_ylabel('Mean reward');ax.grid(alpha=.25)
        ax.set_title(f'{param.capitalize()} sensitivity')
    fig.tight_layout();fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)


def render_gif(env_factory, q, path, fps=3):
    """Find a successful rollout and save frames emitted by env.render()."""
    rollout=successful_rollout(env_factory,q,search_seeds=600)
    if rollout is None:
        raise RuntimeError('No successful greedy rollout found in 600 seeds. Increase SARSA training episodes.')
    # The same reset seed and deterministic chosen actions recreate this exact sequence.
    env=env_factory(render_mode='rgb_array')
    state,_=env.reset(seed=rollout['seed'])
    frames=[env.render()]
    for action in rollout['actions']:
        state,reward,term,trunc,_=env.step(action)
        frames.append(env.render())
        if term or trunc:break
    env.close()
    # Hold the final state long enough to notice success.
    frames.extend([frames[-1]]*4)
    imageio.mimsave(path,frames,duration=1/fps,loop=0)
    return rollout


def _title(c,text,y,size=13):
    c.setFillColor(BLUE);c.setFont('Helvetica-Bold',size);c.drawString(LEFT,y,text)
    return y-19


def _text(c,text,x,y,max_w=495,font='Helvetica',size=9.5,leading=14):
    """A tiny word wrapper that returns the next baseline."""
    c.setFont(font,size);c.setFillColor(DARK)
    lines=[]
    for para in text.split('\n'):
        if not para.strip():lines.append('');continue
        words=para.split();line=''
        for word in words:
            candidate=f'{line} {word}'.strip()
            if stringWidth(candidate,font,size)>max_w and line:
                lines.append(line);line=word
            else: line=candidate
        lines.append(line)
    for line in lines:
        c.drawString(x,y,line);y-=leading
    return y


def _table(c, x, y, widths, rows, font_size=9.0, row_height=25):
    total_w=sum(widths)
    for idx,row in enumerate(rows):
        if idx==0:
            c.setFillColor(LIGHT)
            c.rect(x,y-row_height,total_w,row_height,fill=1,stroke=0)
        elif idx%2==0:
            c.setFillColor(HexColor('#f8fafc'))
            c.rect(x,y-row_height,total_w,row_height,fill=1,stroke=0)
        c.setStrokeColor(HexColor('#d3dee6'))
        c.rect(x,y-row_height,total_w,row_height,fill=0,stroke=1)
        cur=x
        for val,w in zip(row,widths):
            c.setFillColor(DARK)
            c.setFont('Helvetica-Bold' if idx==0 else 'Helvetica',font_size)
            v=str(val)
            while stringWidth(v,'Helvetica',font_size)>w-12 and len(v)>2: v=v[:-2]+'…'
            c.drawString(cur+7,y-row_height+8,v)
            cur+=w
        y-=row_height
    return y


def _page(c,page):
    c.setStrokeColor(HexColor('#cbd7df'))
    c.line(LEFT,39,RIGHT,39)
    c.setFillColor(GRAY);c.setFont('Helvetica',8.5)
    c.drawCentredString(WIDTH/2,27,f'Topics in AI | Assignment 01 | FA23-BAI-044 | Page {page} of 3')
    c.showPage()


def make_report(out_pdf, results, history, sensitivity, plot_files, rollout, meta):
    """PDF is generated from measured metrics; no values are hardcoded."""
    c=canvas.Canvas(str(out_pdf),pagesize=(WIDTH,HEIGHT),pageCompression=1)
    c.setTitle('SARSA Reinforcement Learning Technical Report - Syed M Awais Raza')
    # PAGE ONE
    c.setFillColor(NAVY);c.setFont('Helvetica-Bold',19)
    c.drawString(LEFT,795,'Syed M Awais Raza')
    c.setFont('Helvetica-Bold',11);c.drawString(LEFT,776,'FA23-BAI-044')
    c.drawString(LEFT,753,'Topics in AI | Assignment No. I (CLO-I)')
    c.setFont('Helvetica-Bold',15);c.drawString(LEFT,727,'Custom RL Agent: SARSA on Taxi-v3')
    y=695
    y=_title(c,'1. Problem Description',y)
    y=_text(c,'Taxi-v3 is a discrete 5 x 5 transportation task. A taxi must collect a waiting passenger and deliver the passenger to the indicated goal. The observation is an integer representing 500 possible states (5 rows x 5 columns x 5 passenger statuses x 4 destinations). Six actions are available: south, north, east, west, pickup, and dropoff.',LEFT,y)
    y-=8
    y=_text(c,'Rewards: -1 per ordinary step, -10 for illegal pickup/dropoff, and +20 for successful delivery. An episode ends at delivery or after 200 steps. The objective is to maximize cumulative reward by planning efficient trips.',LEFT,y)
    y-=20;y=_title(c,'2. Methodology: On-policy SARSA',y)
    y=_text(c,'Unlike Q-Learning, SARSA estimates the action value of the action actually chosen under the epsilon-greedy behavior policy. Its temporal-difference update is:',LEFT,y)
    y-=6
    c.setFillColor(NAVY);c.setFont('Courier-Bold',10)
    c.drawString(LEFT+17,y,'Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s_next,a_next) - Q(s,a)]')
    y-=25
    y=_text(c,'The next action is selected by the same epsilon-greedy policy before updating the current state-action pair. True terminal transitions use only the immediate reward; time-limit truncated transitions may still bootstrap. Epsilon decays linearly for a balance of exploration and exploitation. Values are stored in a 500 x 6 NumPy table.',LEFT,y)
    y-=18;y=_title(c,'3. Experimental Setup',y)
    cfg=meta['config']
    settings=[['Setting','Value','Setting','Value'],['Environment','Taxi-v3','Algorithm','Tabular SARSA'],
              ['Episodes',str(cfg['episodes']),'Learning rate',str(cfg['alpha'])],
              ['Gamma',str(cfg['gamma']),'Max steps',str(cfg['max_steps'])],
              ['Initial epsilon',str(cfg['epsilon_start']),'Final epsilon',str(cfg['epsilon_end'])],
              ['Decay fraction',str(cfg['decay_fraction']),'Random seed',str(cfg['seed'])],
              ['Evaluation episodes',str(meta['eval_episodes']),'Q-table','500 x 6']]
    y=_table(c,LEFT,y,[140,105,142,108],settings,row_height=22,font_size=8.5)
    y-=16
    y=_text(c,'Execution provenance: '+meta['backend_label']+'. Evaluation uses identical starting seeds for the trained agent and random baseline. This report is regenerated when main.py runs in Colab.',LEFT,y,size=8.5,leading=13)
    _page(c,1)
    # PAGE TWO
    y=790
    y=_title(c,'4. Results: Training Convergence',y)
    plot=str(plot_files['training'])
    c.drawImage(ImageReader(plot),LEFT+18,367,width=458,height=385,preserveAspectRatio=False,mask='auto')
    y=350
    slice_n=min(250,len(history['rewards']))
    early=np.mean(history['rewards'][:slice_n]);late=np.mean(history['rewards'][-slice_n:])
    y=_text(c,f'Mean training reward: first {slice_n} episodes {early:.2f}; last {slice_n} episodes {late:.2f}. Final rolling success rate: {np.mean(history["successes"][-slice_n:])*100:.1f}%. Curves reveal how the SARSA policy changes as exploration decreases.',LEFT,y,size=9.4)
    y-=20;y=_title(c,'5. Evaluation: Random Policy vs Trained SARSA',y)
    rows=[['Policy','Mean reward','Success rate','Mean steps'],
          ['Random baseline',f'{results["random"]["mean_reward"]:.2f}',f'{results["random"]["success_rate"]*100:.1f}%',f'{results["random"]["mean_steps"]:.1f}'],
          ['Trained SARSA',f'{results["trained"]["mean_reward"]:.2f}',f'{results["trained"]["success_rate"]*100:.1f}%',f'{results["trained"]["mean_steps"]:.1f}']]
    y=_table(c,LEFT,y,[165,120,110,100],rows,row_height=27)
    y-=16
    y=_text(c,'Both policies were evaluated for '+str(meta['eval_episodes'])+' episodes with the same reset seed sequence. Greedy evaluation disables epsilon exploration; a success is a legally completed delivery. The plots and figures are from the executed experiments.',LEFT,y)
    _page(c,2)
    # PAGE THREE
    y=790;y=_title(c,'6. Hyperparameter Sensitivity',y)
    c.drawImage(ImageReader(str(plot_files['sensitivity'])),LEFT+12,566,width=470,height=198,mask='auto')
    y=550
    table_rows=[['Parameter','Value','Mean reward','Success rate']]
    for row in sensitivity:
        table_rows.append([row['parameter'],f'{row["value"]:.2f}',f'{row["mean_reward"]:.2f}',f'{row["success_rate"]*100:.1f}%'])
    y=_table(c,LEFT,y,[135,90,135,135],table_rows,font_size=8.9,row_height=23)
    y-=13
    best=max(sensitivity,key=lambda x:(x['mean_reward'],x['success_rate']))
    y=_text(c,f'Each candidate used {meta["sensitivity_episodes"]} training episodes and {meta["eval_episodes"]} common evaluation seeds. Best tested shorter-run setting: {best["parameter"]}={best["value"]:.2f}, mean reward {best["mean_reward"]:.2f}, success {best["success_rate"]*100:.1f}%. These shorter runs are not a substitute for the full experiment.',LEFT,y,size=8.6,leading=12)
    y-=17;y=_title(c,'7. Discussion and Failure Modes',y)
    y=_text(c,'Early SARSA episodes often wander and incur illegal pickup or dropoff penalties. With enough visits, the learned values can prefer legal and shorter routes. The on-policy target incorporates the next exploratory action, which can make SARSA more conservative than off-policy Q-Learning during training. Weaknesses include incomplete state coverage, slow convergence, and sensitivity to exploration decay and learning rate.',LEFT,y,size=9,leading=13)
    y-=16;y=_title(c,'8. Conclusion',y)
    statement=(f'SARSA achieved {results["trained"]["success_rate"]*100:.1f}% successful deliveries with mean reward '
               f'{results["trained"]["mean_reward"]:.2f} in {meta["eval_episodes"]} evaluation episodes. '
               f'The random policy achieved {results["random"]["success_rate"]*100:.1f}% with mean reward '
               f'{results["random"]["mean_reward"]:.2f}. ')
    if rollout:
        statement+=f'A successful demo required {rollout["steps"]} steps. '
    statement+='The Q-table, training logs and animation enable reproducibility. Re-run the notebook in Google Colab to produce official Gymnasium outputs.'
    y=_text(c,statement,LEFT,y,size=9,leading=13)
    _page(c,3)
    c.save()
