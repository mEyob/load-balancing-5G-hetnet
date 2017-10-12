#!/usr/bin/env WolframScript -script

t = $ScriptCommandLine
t = Take[t, {3,-4}]
t = Partition[t, Length[t]/2]
Print[ToString[t[[1]]]<>ToString[1]]
Print[ToString[t[[2]]]<>ToString[2]]
Exit[];
