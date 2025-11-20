# Simulación Computacional de un Imán Cayendo a Través de un Tubo Conductor Discretizado en Anillos

Este repositorio contiene un simulador físico–computacional que modela la caída de un imán cilíndrico magnetizado a través de un tubo conductor.  
El tubo se discretiza como un conjunto de **N anillos conductores independientes**, cada uno con respuesta electromagnética local.

El fenómeno físico se basa en **inducción electromagnética, corrientes de Foucault (eddy currents)** y **frenado electromagnético** resultante.

---

## Objetivos

- Determinar la dinámica (posición, velocidad y aceleración) del imán durante su caída.
- Calcular las **corrientes inducidas** en cada uno de los anillos.
- Analizar la influencia de distintos parámetros físicos y geométricos.
- Comparar el modelo discreto con posibles experimentos educativos de baja complejidad.

---

## Fundamento Teórico

### Ley de Faraday

Cuando el imán se desplaza, produce un cambio en el flujo magnético \(\Phi_i\) a través de cada anillo:

\[
\mathcal{E}_i = -\frac{d\Phi_i}{dt}
\]

donde:

\[
\Phi_i(z_m) = \int_0^{a} B_z(r, z_i)\,2\pi r\,dr
\]

- \(z_m(t)\): posición del imán
- \(z_i\): posición fija del anillo \(i\)
- \(a\): radio del anillo

---

### Ley de Lenz + Circuito RL

Cada anillo se modela como un circuito RL independiente:

\[
L_i \frac{dI_i}{dt} + R_i I_i = \mathcal{E}_i
\]

con inductancia aproximada:

\[
L_i \approx \mu_0 a\left[\ln\left(\frac{8a}{\rho}\right) - 2\right]
\]

donde \(\rho\) es el radio del alambre.

---

### Extensión a N Anillos — Modelo Matricial

Sea el vector de corrientes:

\[
\mathbf{I}(t) = [I_1(t), I_2(t), \dots , I_N(t)]^{T}
\]

y el vector de fem inducidas:

\[
\mathbf{E}(t) = -\frac{d}{dt}\mathbf{\Phi}(t)
\]

con:

\[
\mathbf{\Phi}(t) = [\Phi_1(t), \Phi_2(t), \dots , \Phi_N(t)]^{T}
\]

Para el caso simple **sin inductancia mutua** (modelo inicial del proyecto):

\[
\mathbf{L}\,\frac{d\mathbf{I}}{dt} + \mathbf{R}\,\mathbf{I} = \mathbf{E}
\]

donde:

\[
\mathbf{L} = \mathrm{diag}(L_1, L_2, \dots, L_N), \quad
\mathbf{R} = \mathrm{diag}(R_1, R_2, \dots, R_N)
\]

---

### Fuerza Magnética Sobre el Imán

La fuerza inducida por cada anillo es:

\[
F_{z,i} \approx m\,\frac{\partial B_{z,i}}{\partial z}
\]

y la fuerza total:

\[
F_z = \sum_{i=1}^{N} F_{z,i}
\]

por lo que la ecuación de movimiento final es:

\[
M\frac{d^2 z_m}{dt^2} = F_z - Mg
\]

---

## Diagrama General del Algoritmo

1. Inicializar posición, velocidad y corrientes.
2. Para cada paso temporal:
   - Calcular flujo magnético de cada anillo.
   - Obtener fem mediante diferencias finitas.
   - Resolver sistema RL.
   - Calcular campos inducidos y fuerza neta.
   - Integrar dinámica del imán.
3. Guardar y graficar resultados.

---

## Resultados Esperados

- Frenado electromagnético creciente con más anillos.
- Corrientes máximas cuando el imán está alineado con cada anillo.
- Velocidad terminal menor que caída libre.
- Comportamiento no lineal dependiente del gradiente de \(\vec{B}\).

---



##  Bibliografía y Artículos 

- Griffiths, D. J. **Introduction to Electrodynamics**, 4th Ed. Cambridge University Press, 2017.
- Jackson, J. D. **Classical Electrodynamics**, 3rd Ed., Wiley, 1999.
- Heald, M. A. “Magnetic braking: improved theory”, *American Journal of Physics*, **56**, 521–522, 1988.
- Saslow, W. M., “Maxwell’s theory of eddy currents in thin conducting sheets…”, *Am. J. Phys.*, **60**, 693–711, 1992.
- Babic, S., & Akyel, C. **"New analytical solution for mutual inductance of coaxial circular coils"**, *IEEE Trans. Magn.*, 44(4), 2008.

---



