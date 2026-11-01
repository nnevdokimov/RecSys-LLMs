"""Content cosine and directional singleton association rules, fitted pre-cutoff."""
import re
import time
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


def top_five(scores, candidates, popularity, k=5):
    """Positive scores first, deterministic popularity/ID ties; common fallback."""
    candidates = np.asarray(candidates)
    scores = np.asarray(scores)
    assert len(candidates) == len(scores) and len(candidates) >= k
    assert np.isfinite(scores).all()
    order = np.lexsort((candidates, -popularity, -np.maximum(scores, 0)))
    return [int(x) for x in candidates[order[:k]]]


def feature_dict(row):
    out = {}
    for field, weight in [('tv_show_category',1),('tv_show_genre_1',2),('tv_show_genre_2',2),('tv_show_genre_3',2),('director',2),('actors',1.5),('year_of_production',0.5)]:
        value = row.get(field)
        if value:
            # Keep whole categorical labels; tokenize people separated by commas/semicolons.
            values = re.split(r'[,;]', value) if field in ('actors','director') else [value]
            prefix = 'genre' if field.startswith('tv_show_genre') else field
            for v in values:
                token = v.strip().casefold()
                if token:
                    out[prefix+'='+token] = weight
    for token in set(re.findall(r'\w{2,}', (row.get('tv_show_title') or '').casefold())):
        out['title='+token] = 1.0
    return out


class Models:
    def __init__(self, train_rows, metadata, users, candidates, cutoff, target_users=None):
        self.users = list(users)
        self.candidates = np.array(sorted(candidates), dtype=np.int64)
        self.items = sorted(set(metadata) | {int(r['item']) for r in train_rows} | set(candidates))
        self.ix = {i:j for j,i in enumerate(self.items)}
        self.ux = {u:j for j,u in enumerate(self.users)}
        self.target_users = list(target_users if target_users is not None else users)
        self.tx = np.array([self.ux[u] for u in self.target_users])
        self.cx = np.array([self.ix[int(i)] for i in self.candidates])
        self.metadata = metadata
        self.rows = train_rows
        self.cutoff = cutoff
        rr, cc, vv = [], [], []
        for r in train_rows:
            if r['user_id'] in self.ux:
                rr.append(self.ux[r['user_id']]); cc.append(self.ix[int(r['item'])]); vv.append(r['n'])
        self.counts = sparse.csr_matrix((vv,(rr,cc)),shape=(len(users),len(self.items)),dtype=np.float64)
        self.popularity = np.asarray((self.counts > 0).sum(axis=0)).ravel()
        self.pop = self.popularity[self.cx]
        self.history_n = np.asarray(self.counts.sum(axis=1)).ravel()
        self.history_items = np.diff(self.counts.indptr)
        self.times = {}

    def weights(self, half_life=None):
        if half_life is None:
            return self.counts
        assert half_life == 28, 'Cached per-airing decay is defined for a 28-day half-life'
        rr,cc,vv=[],[],[]
        for r in self.rows:
            if r['user_id'] in self.ux:
                rr.append(self.ux[r['user_id']]);cc.append(self.ix[int(r['item'])])
                vv.append(r['n28'])
        return sparse.csr_matrix((vv,(rr,cc)),shape=self.counts.shape)

    def popularity_scores(self):
        return np.broadcast_to(self.pop,(len(self.target_users),len(self.candidates)))

    def repeat(self, half_life=None):
        return self.weights(half_life)[self.tx][:,self.cx].toarray()

    def content(self, half_life=None):
        t0=time.perf_counter()
        # Vocabulary and IDF exclusively from metadata known before cutoff.
        known = sorted(self.metadata)
        vec=DictVectorizer(dtype=np.float64)
        x0=vec.fit_transform([feature_dict(self.metadata[i]) for i in known])
        tf=TfidfTransformer(norm=None,sublinear_tf=False)
        tf.fit(x0)
        x=normalize(tf.transform(vec.transform([feature_dict(self.metadata.get(i,{})) for i in self.items])),norm='l2')
        profile=normalize(self.weights(half_life)[self.tx]@x,norm='l2')
        scores=(profile@x[self.cx].T).toarray()
        self.times['content_'+str(half_life)]=time.perf_counter()-t0
        return scores

    def association(self, shrink=10, min_pair=3, neighbors=100, half_life=None):
        t0=time.perf_counter()
        b=(self.counts>0).astype(np.float64)
        co=(b.T@b).tocsr()
        co.setdiag(0);co.eliminate_zeros()
        support=np.asarray(b.sum(axis=0)).ravel()
        # Directional confidence with shrinkage: n(i,j)/(n(i)+lambda).
        for i in range(len(self.items)):
            a,z=co.indptr[i:i+2]
            vals=co.data[a:z]
            vals[vals<min_pair]=0
            vals /= support[i]+shrink
            if len(vals)>neighbors:
                keep=np.argsort(-vals,kind='stable')[:neighbors]
                mask=np.ones(len(vals),dtype=bool);mask[keep]=False;vals[mask]=0
        co.eliminate_zeros()
        w=self.weights(half_life)[self.tx].copy()
        # Multiple viewings strengthen the antecedent, without linear domination.
        w.data=np.sqrt(w.data)
        scores=(w@co[:,self.cx]).toarray()
        self.times['association_'+str(shrink)+'_'+str(half_life)]=time.perf_counter()-t0
        return scores

    def predict(self, scores):
        return {u:top_five(scores[j], self.candidates,self.pop) for j,u in enumerate(self.target_users)}


def row_max_norm(x):
    den=np.max(x,axis=1,keepdims=True)
    return np.divide(x,den,out=np.zeros_like(x),where=den>0)


def fusion(repeat, association, content, repeat_weight=0.8, ar_weight=0.2, cbf_weight=0):
    return repeat_weight*row_max_norm(repeat)+ar_weight*row_max_norm(association)+cbf_weight*row_max_norm(content)
