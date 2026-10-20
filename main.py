"""Run and export the full SARSA Taxi-v3 assignment.

Examples:
    python main.py                      # official Gymnasium environment
    python main.py --backend reference  # offline preview, clearly labeled
    python main.py --episodes 2000 --sensitivity-episodes 500
"""
import argparse
import json
from pathlib import Path
from dataclasses import asdict, replace
import numpy as np
import pandas as pd

from agent import SARSACfg, train_sarsa, evaluate
from utils import plot_training, plot_comparison, plot_sensitivity, render_gif, make_report

STUDENT = 'Syed M Awais Raza'
REGISTRATION = 'FA23-BAI-044'


def get_environment(backend, max_steps):
    if backend == 'reference':
        from taxi_reference import TaxiReference
        return lambda render_mode=None: TaxiReference(render_mode=render_mode)
    import gymnasium as gym
    # Gymnasium gives a 500-state Taxi-v3 with exactly six discrete actions.
    return lambda render_mode=None: gym.make('Taxi-v3', render_mode=render_mode,
                                             max_episode_steps=max_steps)


def run(episodes=20000, sensitivity_episodes=4000, eval_episodes=100,
        alpha=0.20, gamma=0.99, seed=44, backend='gymnasium', output_dir='outputs'):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=SARSACfg(episodes=episodes,alpha=alpha,gamma=gamma,seed=seed)
    factory=get_environment(backend,cfg.max_steps)
    backend_label = ('Gymnasium Taxi-v3 (official)' if backend=='gymnasium' else
                     'Locally executed Taxi-compatible reference model (NOT official Gymnasium)')
    print(f'\n{STUDENT} | {REGISTRATION}\nAlgorithm: SARSA | {backend_label}')
    q, hist=train_sarsa(factory,cfg)
    np.save(out/'sarsa_q_table.npy',q)
    pd.DataFrame({'episode':np.arange(1,episodes+1),**hist}).to_csv(out/'training_history.csv',index=False)
    random_score=evaluate(factory,episodes=eval_episodes,seed=70000,random_policy=True)
    trained_score=evaluate(factory,q=q,episodes=eval_episodes,seed=70000)
    print('Baseline:',random_score)
    print('SARSA:',trained_score)
    cases = [('alpha',x) for x in [0.10,0.20,0.45]] + [('gamma',x) for x in [0.80,0.95,0.99]]
    sensitivity=[]
    for param,value in cases:
        hc=replace(cfg, episodes=sensitivity_episodes, **{param:value})
        print(f'\nSensitivity {param}={value:.2f}:')
        hp_q,_=train_sarsa(factory,hc,verbose=False)
        score=evaluate(factory,q=hp_q,episodes=eval_episodes,seed=70000)
        sensitivity.append({'parameter':param,'value':value,
                            'mean_reward':score['mean_reward'],
                            'success_rate':score['success_rate'],
                            'mean_steps':score['mean_steps']})
        print(f' mean reward={score["mean_reward"]:.2f}, success={score["success_rate"]*100:.1f}%')
    pd.DataFrame(sensitivity).to_csv(out/'hyperparameter_sensitivity.csv',index=False)
    plot_files={'training':out/'training_convergence.png',
                'comparison':out/'evaluation_comparison.png',
                'sensitivity':out/'hyperparameter_sensitivity.png'}
    plot_training(hist,plot_files['training'])
    plot_comparison(random_score,trained_score,plot_files['comparison'])
    plot_sensitivity(sensitivity,plot_files['sensitivity'])
    rollout=None
    try:
        rollout=render_gif(factory,q,out/'trained_taxi.gif')
        print('GIF successful demonstration:',rollout)
    except Exception as e:
        print(f'WARNING: GIF not produced: {e}')
    meta={'student':STUDENT,'registration':REGISTRATION,'algorithm':'SARSA',
          'environment':'Taxi-v3','backend':backend,'backend_label':backend_label,
          'config':asdict(cfg),'eval_episodes':eval_episodes,
          'sensitivity_episodes':sensitivity_episodes}
    scores={'random':random_score,'trained':trained_score}
    report={'meta':meta,'evaluation':scores,'sensitivity':sensitivity,'rollout':rollout}
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    make_report(out/'technical_report.pdf',scores,hist,sensitivity,plot_files,rollout,meta)
    print('\nSaved outputs in',out.resolve())
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['gymnasium','reference'],default='gymnasium')
    p.add_argument('--episodes',type=int,default=20000)
    p.add_argument('--sensitivity-episodes',type=int,default=4000)
    p.add_argument('--eval-episodes',type=int,default=100)
    p.add_argument('--alpha',type=float,default=0.20)
    p.add_argument('--gamma',type=float,default=0.99)
    p.add_argument('--seed',type=int,default=44)
    p.add_argument('--output-dir',default='outputs')
    a=p.parse_args()
    run(**vars(a))
