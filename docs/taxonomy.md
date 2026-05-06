# Curated term taxonomy (284 terms across 6 categories)

Mirrors Appendix A of the paper. Used for combinatorial PubMed retrieval and
for grounding the LLM extraction prompts.

| # | Category | Terms | Source |
|---|---|---:|---|
| 1 | Seizure Classification | 41 | ILAE Classification of Seizure Types Expanded Version 2017 |
| 2 | Epilepsy Syndromes | 50 | ILAE Classification and Definition of Epilepsy Syndromes 2017 |
| 3 | Focal Epilepsy Localisation – Semiology | 82 | Localisation in Focal Epilepsy: A Practical Guide 2021 |
| 4 | EEG Findings | 39 | EEG Normal Waveforms 2023 + EEG Abnormal Waveforms 2023 |
| 5 | MRI Brain Findings | 44 | MRI-identified Pathology in Adults with New-Onset Seizures 2013 |
| 6 | Anti-Seizure Medication Spectrum | 28 | Efficacy and Tolerability of Antiseizure Drugs 2021 |
| | **Total** | **284** | |

## 1. Seizure Classification (41)

```
Focal Onset
  Consciousness: Aware | Impaired awareness
  Motor: Automatisms | Atonic | Clonic | Epileptic spasms |
         Hyperkinetic | Myoclonic | Tonic
  Non-motor: Autonomic | Behaviour arrest | Cognitive | Emotional | Sensory
  Focal to bilateral tonic-clonic
Generalised Onset
  Motor: Tonic-clonic | Clonic | Tonic | Myoclonic |
         Myoclonic-tonic-clonic | Myoclonic-atonic | Atonic | Epileptic spasms
  Non-motor (absence): Typical | Atypical | Myoclonic | Eyelid myoclonia
Unknown
  Motor: Tonic-clonic | Epileptic spasms
  Non-motor: Behaviour arrest
Unclassified
```

## 2. Epilepsy Syndromes (50)

```
Genetic Generalised Epilepsies (GGE)
  Idiopathic Generalised Epilepsies (IGE):
    Juvenile myoclonic epilepsy (JME) | Juvenile absence epilepsy (JAE) |
    Epilepsy with generalised tonic-clonic seizures alone (GTCA) |
    Childhood absence epilepsy (CAE) |
    Epilepsy with eyelid myoclonia (Jeavons Syndrome) |
    Epilepsy with myoclonic absence (Bureau and Tassinari Syndrome)
Focal Epilepsies
  Self-limited:
    Childhood occipital visual epilepsy (COVE) |
    Photosensitive occipital lobe epilepsy (POLE) | SeLECTS | SeLEAS |
    SeLNE | SeLFNIE | SeLIE | GEFS+ | MEI
  Non-self-limited:
    Familial mesial temporal lobe epilepsy (FMTLE) |
    Epilepsy with auditory features (EAF) | MTLE-HS |
    Sleep related hypermotor epilepsy (SHE) | FFEVF
Developmental and/or Epileptic Encephalopathies (DEE):
  FIRES | Rasmussen syndrome (RS) | Progressive myoclonus epilepsies (PME) |
  EMAtS (Doose Syndrome) | Lennox-Gastaut Syndrome (LGS) | DEE-SWAS | EE-SWAS |
  Landau-Kleffner Syndrome | HHE | EIDEE | EIMFS |
  Infantile epileptic spasms syndrome (IESS) | Dravet Syndrome (DS)
Combined generalised and focal epilepsy syndromes:
  Epilepsy with reading induced seizures (EwRIS)
Aetiology-specific syndromes:
  KCNQ2-DEE | PD-DEE | P5PD-DEE | CDKL5-DEE | PCDH19 clustering epilepsy |
  GLUT1DS | Sturge Weber Syndrome (SWS) |
  Gelastic seizures with hypothalamic hamartoma (GS-HH)
```

## 3. Focal Epilepsy Localisation – Semiology (82)

```
Frontal
  Primary motor cortex
    Lateral: Contralateral leg clonic/tonic activity
    Medial: Contralateral face/arm clonic/tonic activity
  Supplementary motor area: Asymmetric bilateral tonic posturing |
    Extension contralateral upper limb | Flexion ipsilateral upper limb |
    Contralateral head and eye deviation
  Dorsolateral prefrontal cortex / orbitofrontal:
    Complex automatisms | Dialeptic | Semipurposeful behaviour
  Frontal eyefields: Contralateral head/eye version
  Mesial prefrontal / anterior cingulate:
    Hyperkinetic | Ictal fear | Vocalisation | Chapeau de gendarme
  Broca's area: Dysphasia
  Frontal operculum: Hypersalivation | Dysarthria | Face clonic activity
  Hypothalamus: Gelastic
Temporal
  Mesial temporal including hippocampus:
    Behavioural arrest | Manual automatisms | Oroalimentary automatisms |
    Gustatory aura | Psychic aura | Ictal speech (non-dominant) |
    Contralateral dystonic posturing
  Amygdala: Early autonomic signs (tachycardia, apnoea)
  Insula
    Anterior: Viscerosensory symptoms | Autonomic symptoms
    Posterior: Painful sensations (burning, tingling)
  Lateral temporal / neocortical: Auditory aura
Parietal
  Primary sensory cortex: Contralateral numbness/paraesthesia |
    Contralateral pain or altered thermal sensation
  Parietal association area
    Superior parietal lobule and precuneus:
      Macrosomatagnosis | Microsomatagnosis | Kinetopsia
    Precuneus: Macropsia | Micropsia
  Temporoparietal junction:
    Vertigo | Autoscopy | Language impairment (dominant)
  Secondary sensory area: Inability to move contralateral arm
Occipital
  Primary visual cortex:
    Flashing coloured or bright lights (usually contralateral)
  Visual association areas:
    Complex visual hallucinations | Kinetopsia | Macropsia | Micropsia | Autoscopy
  Parieto-occipital junction: Eye version | Nystagmus | Blinking
```

## 4. EEG Findings (39)

```
Normal
  Non-epileptiform transients: Lambda waves | POSTS | Phantom spike and wave |
    Ctenoids | Vertex Sharp Transients | K Complexes | BETS | Wicket waves | RMTD
Abnormal
  Epileptiform discharges
    Morphology: Sharp | Spike and wave
    Patterns: 3 Hz spike and wave | Centro-temporal (Rolandic) spikes |
      Continuous spike and wave during sleep | Slow spike and wave |
      Polyspike and wave | Generalised spike and wave |
      Lateralising periodic discharges | Bilateral independent periodic discharges |
      Generalised periodic discharges | Subclinical EEG discharges of adults |
      Brief potentially ictal rhythmic epileptiform discharges
  Non-epileptiform
    Slowing: Focal slowing | Frontal IRDA | Occipital IRDA | Temporal IRDA |
      Diffuse slowing
    Triphasic waves | Breach rhythm | Burst suppression | Electrocerebral inactivity
```

## 5. MRI Brain Findings (44)

```
Epileptogenic lesion
  Gliosis/encephalomalacia: Ulegyria | Poststroke | Posttraumatic |
    Postoperative | Unspecified
  Mesial temporal sclerosis | Encephalocoele
  Developmental abnormality: Grey matter heterotopia | Focal cortical dysplasia |
    Polymicrogyria | Other types of cortical malformation | Tuberous sclerosis |
    Nodular heterotopia | Band heterotopia | Subdural cystic hygroma |
    Perinatal ischaemic lesion
  Vascular anomaly: Cavernoma | Arteriovenous malformation
  Tumors: Glioma | Nonspecific tumor mass
  Neurocysticercosis | Granulomatous inflammation
Non-epileptogenic abnormality
  Nonspecific white matter / T2 hyperintensity |
  Cerebellar volume loss and atrophy | Cerebral volume loss and atrophy |
  Small vessel disease | Nonspecific small infarcts | Small granuloma |
  Corpus callosum agenesis | Developmental venous anomaly |
  Hippocampal structures asymmetry | Chiari I malformation |
  Choroidal fissure cyst | Cystic lesion | Leukodystrophy | Local dystrophy |
  Nonspecific nodule | Rathke's cleft cyst
```

## 6. Anti-Seizure Medication Spectrum (28)

```
Brivaracetam | Cannabidiol | Carbamazepine | Cenobamate | Clobazam |
Clonazepam | Eslicarbazepine | Ethosuximide | Felbamate | Fenfluramine |
Gabapentin | Lacosamide | Lamotrigine | Levetiracetam | Oxcarbazepine |
Perampanel | Phenobarbital | Phenytoin | Pregabalin | Primidone |
Retigabine | Rufinamide | Stiripentol | Tiagabine | Topiramate |
Valproate | Vigabatrin | Zonisamide
```
