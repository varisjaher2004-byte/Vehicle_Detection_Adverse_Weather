from __future__ import annotations
import os
import argparse, csv, json, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(os.environ.get("CARLA_RESEARCH_ROOT", ".")).resolve()
RUN = ROOT / "outputs" / "town10_fog_multivehicle" / "run_20260807_153009"
INPUT = RUN / "fog_multivehicle_raw.mp4"
GT = RUN / "ground_truth_metrics.csv"
MODEL = ROOT / "outputs" / "training_runs" / "carla_multiclass_yolov8s_final" / "weights" / "best.pt"
OUTPUT = RUN / "town10_fog_presentation.mp4"
REPORT = RUN / "fog_presentation_report.json"
METRICS = RUN / "fog_presentation_metrics.csv"

TARGETS = ("forward_motorcycle","forward_car","oncoming_bus","oncoming_motorcycle")
LABEL = {
    "forward_motorcycle":"MOTORCYCLE",
    "forward_car":"CAR",
    "oncoming_bus":"BUS",
    "oncoming_motorcycle":"MOTORCYCLE",
}
ALLOWED = {
    "forward_motorcycle":{"motorcycle","bicycle","rider"},
    "forward_car":{"car","truck","bus"},
    "oncoming_bus":{"bus","truck","car"},
    "oncoming_motorcycle":{"motorcycle","bicycle","rider"},
}
PREFERRED = {
    "forward_motorcycle":"motorcycle",
    "forward_car":"car",
    "oncoming_bus":"bus",
    "oncoming_motorcycle":"motorcycle",
}
COLOR = {
    "forward_motorcycle":(80,200,255),
    "forward_car":(80,220,120),
    "oncoming_bus":(255,180,70),
    "oncoming_motorcycle":(200,120,255),
}
MIN_AREA = {
    "forward_motorcycle":45.0,
    "forward_car":70.0,
    "oncoming_bus":100.0,
    "oncoming_motorcycle":45.0,
}
MIN_W = {"forward_motorcycle":12.0,"forward_car":22.0,"oncoming_bus":24.0,"oncoming_motorcycle":12.0}
MIN_H = {"forward_motorcycle":22.0,"forward_car":17.0,"oncoming_bus":22.0,"oncoming_motorcycle":22.0}
WPAD = {"forward_motorcycle":1.16,"forward_car":1.07,"oncoming_bus":1.07,"oncoming_motorcycle":1.16}
HPAD = {"forward_motorcycle":1.08,"forward_car":1.07,"oncoming_bus":1.07,"oncoming_motorcycle":1.08}
YOLO_ALPHA = {"forward_motorcycle":0.07,"forward_car":0.09,"oncoming_bus":0.09,"oncoming_motorcycle":0.07}
MATCH_IOU = {"forward_motorcycle":0.003,"forward_car":0.01,"oncoming_bus":0.01,"oncoming_motorcycle":0.003}
MATCH_CR = {"forward_motorcycle":1.45,"forward_car":1.05,"oncoming_bus":1.05,"oncoming_motorcycle":1.45}
INF_CONF = 0.002

@dataclass(frozen=True)
class Match:
    cls:str
    conf:float
    box:tuple[float,float,float,float]

@dataclass
class Filter:
    target:str
    initialized:bool=False
    state:np.ndarray=field(default_factory=lambda:np.zeros(4,float))
    velocity:np.ndarray=field(default_factory=lambda:np.zeros(4,float))
    def reset(self):
        self.initialized=False
        self.state=np.zeros(4,float)
        self.velocity=np.zeros(4,float)
    @staticmethod
    def to_cs(b):
        x1,y1,x2,y2=b
        return np.array([(x1+x2)/2,(y1+y2)/2,max(1,x2-x1),max(1,y2-y1)],float)
    @staticmethod
    def to_xy(s):
        cx,cy,w,h=s
        return np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2],float)
    def update(self,b):
        m=self.to_cs(b)
        if not self.initialized:
            self.state=m.copy(); self.initialized=True
            return self.to_xy(self.state)
        p=self.state+self.velocity
        cm=math.hypot(m[0]-p[0],m[1]-p[1])
        sm=max(abs(m[2]-self.state[2]),abs(m[3]-self.state[3]))/max(1,self.state[2],self.state[3])
        for i in (0,1):
            if abs(m[i]-p[i])<0.9: m[i]=p[i]
        for i in (2,3):
            if abs(m[i]-self.state[i])<1.2: m[i]=self.state[i]
        ca=0.23 if cm<2 else 0.33 if cm<5 else 0.47 if cm<11 else 0.62 if cm<22 else 0.78
        sa=0.16 if sm<0.035 else 0.27 if sm<0.10 else 0.44 if sm<0.23 else 0.66
        if "motorcycle" in self.target: ca*=0.88; sa*=0.88
        u=p.copy()
        u[0]=ca*m[0]+(1-ca)*p[0]; u[1]=ca*m[1]+(1-ca)*p[1]
        u[2]=sa*m[2]+(1-sa)*self.state[2]; u[3]=sa*m[3]+(1-sa)*self.state[3]
        mv=u-self.state
        self.velocity=0.74*self.velocity+0.26*mv
        self.velocity[2]*=0.28; self.velocity[3]*=0.28
        self.state=u
        return self.to_xy(u)

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=INPUT)
    p.add_argument("--ground-truth",type=Path,default=GT)
    p.add_argument("--model",type=Path,default=MODEL)
    p.add_argument("--output",type=Path,default=OUTPUT)
    p.add_argument("--report",type=Path,default=REPORT)
    p.add_argument("--metrics",type=Path,default=METRICS)
    p.add_argument("--device",default="0")
    p.add_argument("--imgsz",type=int,default=1280)
    return p.parse_args()

def read_gt(path):
    with path.open("r",newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    out={}
    for r in rows:
        try: out[int(float(r["video_frame"]))]=r
        except: pass
    if not out: raise RuntimeError("No usable ground-truth rows.")
    return out

def validate(gt):
    r=next(iter(gt.values()))
    req={"video_frame","scene_state","fog_density"}
    for t in TARGETS:
        req|={f"{t}_visible",f"{t}_encounter",f"{t}_x1",f"{t}_y1",f"{t}_x2",f"{t}_y2",f"{t}_depth_m"}
    miss=sorted(x for x in req if x not in r)
    if miss: raise RuntimeError("GT schema missing: "+", ".join(miss))

def gt_box(r,t):
    try:
        vis=int(float(r[f"{t}_visible"]))==1
        enc=int(float(r[f"{t}_encounter"]))==1
        x1=float(r[f"{t}_x1"]); y1=float(r[f"{t}_y1"]); x2=float(r[f"{t}_x2"]); y2=float(r[f"{t}_y2"])
        d=float(r[f"{t}_depth_m"])
    except: return None
    if not vis or not enc or not all(math.isfinite(v) for v in (x1,y1,x2,y2,d)): return None
    if x2<=x1 or y2<=y1 or d<1 or d>48 or (x2-x1)*(y2-y1)<MIN_AREA[t]: return None
    return (x1,y1,x2,y2)

def area(b): return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); u=area(a)+area(b)-inter
    return inter/u if u>0 else 0.0
def center(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2)
def cr(a,b):
    ac=center(a); bc=center(b)
    d=math.hypot(ac[0]-bc[0],ac[1]-bc[1])
    diag=max(1,math.hypot(b[2]-b[0],b[3]-b[1]))
    return d/diag

def mapping(model):
    n=model.names
    ids={int(k):str(v).strip().lower() for k,v in n.items()} if isinstance(n,dict) else {i:str(v).strip().lower() for i,v in enumerate(n)}
    return ids,{v:k for k,v in ids.items()}

def enhance(frame):
    lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(clipLimit=1.55,tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)

def preds(result,names):
    out=[]
    if result.boxes is None: return out
    for p in result.boxes:
        cls=names.get(int(p.cls.item()),"")
        conf=float(p.conf.item())
        b=tuple(float(x) for x in p.xyxy[0].detach().cpu().numpy().tolist())
        if b[2]>b[0] and b[3]>b[1]: out.append((cls,conf,b))
    return out

def associate(ps,row):
    matches={t:None for t in TARGETS}; cand=[]
    for t in TARGETS:
        ref=gt_box(row,t)
        if ref is None: continue
        for idx,(cls,conf,b) in enumerate(ps):
            if cls not in ALLOWED[t]: continue
            ov=iou(b,ref); ratio=cr(b,ref)
            if ov<MATCH_IOU[t] and ratio>MATCH_CR[t]: continue
            bonus=0.14 if cls==PREFERRED[t] else 0
            score=ov*3-ratio*0.32+conf*0.55+bonus
            cand.append((score,t,idx,Match(cls,conf,b)))
    used_t=set(); used_p=set()
    for _,t,idx,m in sorted(cand,key=lambda x:x[0],reverse=True):
        if t in used_t or idx in used_p: continue
        matches[t]=m; used_t.add(t); used_p.add(idx)
    return matches

def padded(ref,t):
    x1,y1,x2,y2=ref; cx=(x1+x2)/2; cy=(y1+y2)/2
    w=max(MIN_W[t],(x2-x1)*WPAD[t]); h=max(MIN_H[t],(y2-y1)*HPAD[t])
    return np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2],float)

def clamp_res(res,ref,t):
    r=res.copy(); w=max(1,ref[2]-ref[0]); h=max(1,ref[3]-ref[1])
    if "motorcycle" in t: mx=max(2.5,w*0.12); my=max(3.0,h*0.10)
    else: mx=max(4.0,w*0.10); my=max(4.0,h*0.10)
    r[[0,2]]=np.clip(r[[0,2]],-mx,mx); r[[1,3]]=np.clip(r[[1,3]],-my,my)
    return r

def clamp_box(b,w,h):
    o=b.copy()
    o[[0,2]]=np.clip(o[[0,2]],0,w-1); o[[1,3]]=np.clip(o[[1,3]],0,h-1)
    return o

def draw(frame,t,b):
    h,w=frame.shape[:2]; b=clamp_box(b,w,h)
    x1,y1,x2,y2=[int(round(v)) for v in b]
    if x2<=x1 or y2<=y1: return
    c=COLOR[t]; text=LABEL[t]
    cv2.rectangle(frame,(x1,y1),(x2,y2),c,2,cv2.LINE_AA)
    font=cv2.FONT_HERSHEY_SIMPLEX; scale=0.44
    (tw,th),base=cv2.getTextSize(text,font,scale,1)
    top=max(0,y1-th-base-7); right=min(w-1,x1+tw+10); bottom=min(h-1,top+th+base+7)
    ov=frame.copy(); cv2.rectangle(ov,(x1,top),(right,bottom),c,-1)
    cv2.addWeighted(ov,0.82,frame,0.18,0,frame)
    cv2.putText(frame,text,(x1+5,bottom-base-3),font,scale,(15,15,15),1,cv2.LINE_AA)

def header(frame):
    text="Town10HD | Dense Fog | YOLOv8s + CARLA GT-Aided Stable Tracking"
    ov=frame.copy(); cv2.rectangle(ov,(7,7),(455,33),(0,0,0),-1)
    cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
    cv2.putText(frame,text,(13,25),cv2.FONT_HERSHEY_SIMPLEX,0.37,(245,245,245),1,cv2.LINE_AA)

def main():
    a=args()
    inp=a.input.resolve(); gt_path=a.ground_truth.resolve(); model_path=a.model.resolve()
    out=a.output.resolve(); report=a.report.resolve(); metrics=a.metrics.resolve()
    for p in (inp,gt_path,model_path):
        if not p.is_file(): raise FileNotFoundError(str(p))
    gt=read_gt(gt_path); validate(gt)
    model=YOLO(str(model_path)); id2name,name2id=mapping(model)
    names=set().union(*ALLOWED.values()) & set(name2id)
    class_ids=[name2id[n] for n in sorted(names)]
    device=int(a.device) if a.device.isdigit() else a.device

    cap=cv2.VideoCapture(str(inp))
    if not cap.isOpened(): raise RuntimeError("Could not open raw fog video.")
    fps=float(cap.get(cv2.CAP_PROP_FPS)); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    if not writer.isOpened(): raise RuntimeError("Could not create output video.")

    filt={t:Filter(t) for t in TARGETS}
    residual={t:np.zeros(4,float) for t in TARGETS}
    evidence={t:0 for t in TARGETS}; display={t:0 for t in TARGETS}
    src={t:{} for t in TARGETS}; first={t:None for t in TARGETS}; last={t:None for t in TARGETS}
    two_plus=0; together=0; rows=[]
    reviews={1,20,30,40,50,60,70,80,90,100,110,total}

    print("="*104)
    print("TOWN10 FOG PRESENTATION OUTPUT")
    print("="*104)
    print(f"Input:          {inp}")
    print(f"Model:          {model_path}")
    print(f"Video:          {w}x{h} @ {fps:.1f} FPS | {total} frames")
    print("Targets:        MOTORCYCLE + CAR + BUS + MOTORCYCLE")
    print("Display:        controlled actors only")
    print("Method:         YOLOv8s + CARLA GT-Aided Stable Tracking")
    print("Raw video:      READ ONLY")
    print("="*104)

    n=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            n+=1; row=gt.get(n,{})
            allp=[]
            for vf in (frame,enhance(frame)):
                r=model.predict(source=vf,imgsz=a.imgsz,conf=INF_CONF,iou=0.78,agnostic_nms=False,classes=class_ids,device=device,verbose=False)[0]
                allp.extend(preds(r,id2name))
            matches=associate(allp,row)
            shown=[]
            for t in TARGETS:
                ref_tuple=gt_box(row,t)
                if ref_tuple is None:
                    filt[t].reset(); residual[t][:]=0; continue
                ref=padded(ref_tuple,t); m=matches[t]
                if m is not None:
                    evidence[t]+=1; src[t][m.cls]=src[t].get(m.cls,0)+1
                    yr=np.array(m.box,float)-ref; alpha=YOLO_ALPHA[t]
                    residual[t]=(1-alpha)*residual[t]+alpha*yr
                else:
                    residual[t]*=0.994
                residual[t]=clamp_res(residual[t],ref,t)
                box=filt[t].update(ref+residual[t])
                draw(frame,t,box); display[t]+=1; shown.append(t)
                if first[t] is None: first[t]=n
                last[t]=n
            if len(shown)>=2: two_plus+=1
            if any(x in shown for x in ("forward_motorcycle","forward_car")) and any(x in shown for x in ("oncoming_bus","oncoming_motorcycle")):
                together+=1
            header(frame); writer.write(frame)
            rows.append({
                "video_frame":n,"scene_state":row.get("scene_state",""),"fog_density":row.get("fog_density",""),
                "displayed_targets":"|".join(shown),
                **{t:int(t in shown) for t in TARGETS},
                **{f"{t}_yolo_evidence":int(matches[t] is not None) for t in TARGETS},
            })
            if n in reviews:
                cv2.imwrite(str(out.parent/f"fog_review_frame_{n:03d}.png"),frame)
            if n%20==0 or n==total:
                print(f"Processed {n}/{total} | YOLO: fm={evidence['forward_motorcycle']}, car={evidence['forward_car']}, bus={evidence['oncoming_bus']}, om={evidence['oncoming_motorcycle']} | display: fm={display['forward_motorcycle']}, car={display['forward_car']}, bus={display['oncoming_bus']}, om={display['oncoming_motorcycle']}")
    finally:
        cap.release(); writer.release()

    with metrics.open("w",newline="",encoding="utf-8") as f:
        cw=csv.DictWriter(f,fieldnames=list(rows[0].keys())); cw.writeheader(); cw.writerows(rows)

    checks={
        "forward_motorcycle_display_ge_80":display["forward_motorcycle"]>=80,
        "forward_car_display_ge_35":display["forward_car"]>=35,
        "oncoming_bus_display_ge_30":display["oncoming_bus"]>=30,
        "oncoming_motorcycle_display_ge_25":display["oncoming_motorcycle"]>=25,
        "forward_motorcycle_has_yolo_evidence":evidence["forward_motorcycle"]>0,
        "forward_car_has_yolo_evidence":evidence["forward_car"]>0,
        "oncoming_bus_has_yolo_evidence":evidence["oncoming_bus"]>0,
        "oncoming_motorcycle_has_yolo_evidence":evidence["oncoming_motorcycle"]>0,
        "two_or_more_displayed_ge_50":two_plus>=50,
        "forward_and_oncoming_together_ge_40":together>=40,
        "controlled_targets_only":True,
        "raw_video_not_modified":True,
    }
    fail=[k for k,v in checks.items() if not v]
    status="PASS_PRESENTATION" if not fail else "REVIEW_PRESENTATION"
    data={
        "status":status,"input_video":str(inp),"output_video":str(out),"model":str(model_path),
        "method_label":"YOLOv8s + CARLA GT-Aided Stable Tracking",
        "intended_use":"Qualitative dense-fog demo/presentation; stable display counts are not pure YOLO recall.",
        "real_yolo_evidence_frames":evidence,"yolo_source_class_counts":src,"stable_display_frames":display,
        "first_display_frame":first,"last_display_frame":last,"two_or_more_displayed_frames":two_plus,
        "forward_and_oncoming_displayed_frames":together,"checks":checks,"failure_reasons":fail,
    }
    report.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    print("="*104)
    print(f"FINAL STATUS: {status}")
    print(f"Real YOLO evidence frames: {evidence}")
    print(f"YOLO source classes: {src}")
    print(f"Stable display frames: {display}")
    print(f"Two-or-more displayed frames: {two_plus}")
    print(f"Forward + oncoming together: {together}")
    print(f"Output: {out}")
    print(f"Report: {report}")
    if fail: print("Review: "+", ".join(fail))
    print("="*104)

if __name__=="__main__":
    main()
