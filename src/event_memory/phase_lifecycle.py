"""Sequential P/PH closure from observed hierarchy transitions, never EOF.

A new ordered sibling is the manager's transition decision.  Deterministic
bookkeeping closes predecessors, and a closed episode closes its children.
Evidence bounds remain unchanged; closed_at records confirmation time.
"""
from copy import deepcopy
import math

SEQUENTIAL = 'sequential_phase_lifecycle_v1'
PRESERVE = 'preserve_source'
LIFECYCLES = (SEQUENTIAL, PRESERVE)


def _ordered(nodes):
    # Dict insertion order breaks equal-time ties; do not sort PH10 before PH2.
    return sorted(nodes, key=lambda k: nodes[k]['start'])


def reconcile_lifecycle(phases, episodes, now, previous_phases=None, previous_episodes=None):
    """Return new node maps + first-close transitions without mutating input."""
    if isinstance(now,bool) or not isinstance(now,(int,float)) or not math.isfinite(now):
        raise ValueError('Lifecycle time must be finite')
    previous_phases=previous_phases or {};previous_episodes=previous_episodes or {}
    phases,episodes=deepcopy(phases),deepcopy(episodes);transitions=[]
    owners={}
    for pid,p in episodes.items():
        for phid in p['phase_ids']:
            if phid not in phases or phid in owners:raise ValueError('Phase must have one known parent')
            owners[phid]=pid
    if set(phases)!=set(owners):raise ValueError('Every phase must have a parent')
    for nodes,old,key in [(phases,previous_phases,'event_ids'),(episodes,previous_episodes,'phase_ids')]:
        if not set(old).issubset(nodes):raise ValueError('Lifecycle cannot remove a node')
        for nid,node in nodes.items():
            if node['status'] not in ('open','closed'):raise ValueError('Invalid lifecycle status')
            if not math.isfinite(node['end']) or node['end']>now:raise ValueError('Future node in lifecycle')
            if old.get(nid,{}).get('status')=='closed' and node[key]!=old[nid][key]:
                raise ValueError('Cannot append members to a closed node')
    def close(nodes,old,nid,layer,reason,trigger):
        node=nodes[nid];prior=old.get(nid,{})
        node['status']='closed'
        if prior.get('status')=='closed':
            node['closed_at']=prior.get('closed_at',now)
            node['close_reason']=prior.get('close_reason',reason)
            node['close_trigger_id']=prior.get('close_trigger_id',trigger)
        else:
            node.update(closed_at=float(now),close_reason=reason,close_trigger_id=trigger)
            transitions.append({'layer':layer,'id':nid,'now':float(now),'reason':reason,'trigger_id':trigger})
    pids=_ordered(episodes)
    for i,pid in enumerate(pids):
        p=episodes[pid]
        if previous_episodes.get(pid,{}).get('status')=='closed':
            close(episodes,previous_episodes,pid,'episode','closed_status_latched',pid)
        elif p['status']=='closed':
            close(episodes,previous_episodes,pid,'episode','manager_closed',pid)
        elif i<len(pids)-1:
            close(episodes,previous_episodes,pid,'episode','next_episode',pids[i+1])
    for pid in pids:
        p=episodes[pid];ids=p['phase_ids']
        for i,phid in enumerate(ids):
            ph=phases[phid]
            if previous_phases.get(phid,{}).get('status')=='closed':
                close(phases,previous_phases,phid,'phase','closed_status_latched',phid)
            elif ph['status']=='closed':
                close(phases,previous_phases,phid,'phase','manager_closed',phid)
            elif i<len(ids)-1:
                close(phases,previous_phases,phid,'phase','next_phase',ids[i+1])
            elif p['status']=='closed':
                close(phases,previous_phases,phid,'phase','parent_episode_closed',pid)
    return phases,episodes,transitions


class PhaseLifecycleProjection:
    """Repair historical v5 lifecycle at each causal prefix, with provenance."""
    def __init__(self,policy=SEQUENTIAL):
        if policy not in LIFECYCLES:raise ValueError('Unknown source lifecycle policy')
        self.policy=policy;self.phases={};self.episodes={};self.events={};self.history=[];self.last_now=None

    def project(self,source,now):
        value=deepcopy(source)
        if self.policy==PRESERVE:return value,[]
        if self.last_now is not None and now<self.last_now:raise ValueError('Lifecycle time moved backwards')
        for ph in self.phases.values():
            if ph['status']=='closed':
                for eid in ph['event_ids']:
                    if self.events[eid]!=source['events'].get(eid):
                        raise ValueError('Source event changed after repaired phase close')
        phases,episodes,updates=reconcile_lifecycle(source['phases'],source['episodes'],now,self.phases,self.episodes)
        value['phases']=phases;value['episodes']=episodes
        for key in ('phases','episodes'):
            if key in value.get('context',{}):value['context'][key]=deepcopy(value[key])
        self.phases,self.episodes=deepcopy(phases),deepcopy(episodes)
        self.events=deepcopy(source['events']);self.history.extend(deepcopy(updates));self.last_now=now
        return value,updates
