<CsoundSynthesizer>
<CsOptions>
-odac
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 64
nchnls = 2
0dbfs = 1
seed 0

gaRvbL init 0
gaRvbR init 0
gaDelL init 0
gaDelR init 0

; p3: length, p4: ftable, p5: pitch1, p6: pitch2, p7: amp, p8: pan
instr 1
 krvbSendAmt linseg 0.2, p3*0.8, 0.7, p3*0.2, 0.7
 kdelSendAmt linseg 0.2, p3*0.5, 0.7, p3*0.5, 0.7
 kpan = p8
 kfreq expseg p5, p3*0.2, p5, p3*0.6, p6, p3*0.2, p6
 kamp linseg 0, p3*0.1, 1, p3*0.8, 1, p3*0.1, 0 
 kvib expseg 0.001, p3*0.3, 0.001, p3*0.6, 1, p3*0.1, 1
 klfo oscil 1, 15 * kvib, -1

 ; FM
 ;aMod poscil 20, kfreq * 6.3, p4
 ; sound source
 aTone poscil p7*kamp, kfreq * (1+klfo*kvib*0.2), p4

 ; panning
 aoutl, aoutr pan2 aTone, kpan

 ; sends
 gaRvbL += aoutl * krvbSendAmt
 gaRvbR += aoutr * krvbSendAmt
 gaDelL += aoutl * kdelSendAmt
 gaDelR += aoutr * kdelSendAmt

 ; audio out
 outs aoutl, aoutr
endin

instr 99
 idx = 0
 while idx < p4 do
    ifn ftgen idx + 1, 0, 1024, 7, 0, 1024, 0
    idx += 1
 od
endin



instr 10
 ifblvl = 0.8
 ifco = 9000
 aOutL, aOutR reverbsc gaRvbL, gaRvbR, ifblvl, ifco
 outs aOutL, aOutR
 clear gaRvbL
 clear gaRvbR
endin

instr 11
 adl = 0.2
 aOutL vdelayx gaDelL, adl, 2, 64
 aOutR vdelayx gaDelR, adl, 2, 64
 outs aOutL, aOutR
 clear gaDelL
 clear gaDelR
endin



; ifn ftgen 1, 0, 1024, 7, 0, 1024, 0
; ifn ftgen 2, 0, 1024, 7, 0, 1024, 0
; ifn ftgen 3, 0, 1024, 7, 0, 1024, 0
; ifn ftgen 4, 0, 1024, 7, 0, 1024, 0
; ifn ftgen 5, 0, 1024, 7, 0, 1024, 0

</CsInstruments>
<CsScore>
i10 0 z
i11 0 z
</CsScore>
</CsoundSynthesizer>