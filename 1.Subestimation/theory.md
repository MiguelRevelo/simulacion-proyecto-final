# Teoría: Interacción de Imán-Anillo considerando únicamente efectos radiales (Subestimación del efecto)

## **Objetivo específico:**
Desarrollar un modelo discreto simplificado que describa la interacción entre el campo magnético de un imán y las corrientes de Foucault inducidas en cada anillo. Se simplifica tomando únicamente la componente radial de la fuerza de frenado. El objetivo incluye formular las ecuaciones dinámicas y proponer una estrategia computacional para simular la dinámica. 

Inicialmente, partimos de estudiar el caso más simple, en donde se tiene un único anillo, y en su centro se encuentra un imán, el anillo se encuentra en el eje Z de tal manera que el imán cae verticalmente por este eje, inicialmente el anillo tendrá un radio fijo a y su area es $A= \Pi a^{2}$ 

## Interacción Imán-Anillo: 

<img width="300" height="300" alt="imagen" src="https://github.com/user-attachments/assets/3c3d2719-ff24-4a13-8b0b-f7b395d4683f" />

En este sistema, la interacción electromagnética acoplada resulta de dependencias críticas tanto de las propiedades de los materiales _(conductividad, permeabilidad magnética)_ como de la configuración geométrica _(distancia, simetría)_.

Cuando un imán se aproxima a un anillo conductor, la variación temporal del flujo magnético que atraviesa la superficie del anillo induce corrientes de Foucault _(eddy currents)_ en su interior. Estas corrientes generan campos magnéticos propios que, de acuerdo con la ley de Lenz, se oponen al cambio de flujo que las originó. En consecuencia, pueden producir fuerzas de atracción o repulsión dependiendo de la dirección del movimiento y de la orientación del campo del imán.

En este primer caso, el imán cae a lo largo del eje de un anillo conductor. El anillo, al ser eléctricamente cerrado, responde a la variación del campo magnético $B(r,t)$ generado por el imán _(cuyo eje principal coincide con el eje del anillo)_, induciendo una corriente azimutal.

La dependencia crucial surge debido a que la magnitud del campo que atraviesa el área del anillo varía con la posición $z(t)$ del imán, modificando el flujo magnético debido al acople $Φ(t)$ definido como: 

$$\phi(t)= \int_{s} B(r(z_{\text{imán}}),t).dS$$

y generando la Fuerza Electromotríz descrita según la ley de Faraday:  

$$\epsilon(t)= - \frac{d\phi(t)}{dt}$$

_(Esta geometría cerrada del anillo podría evaluarse experimentalmente en etapas posteriores para analizar la dependencia del fenómeno con la conductividad del material y la distancia entre elementos.)_

Adicionalmente, desde otra perspectiva, este sistema puede modelarse como un circuito R-L con una fuente de excitación externa, en donde son relevantes tres parámetros: 
* Resistencia del material conductor **(R)**
* Autoinductancia del anillo **(L)**
* Fuerza Electromotríz inducida $\epsilon(t)$ debido al cambio de flujo magnético que atraviesa el anillo.

La ecuación diferencial que gobierna la dinámica de la corriente inducida $i(t)$ se obtiene combinando la ley de Faraday con la ley de Ohm para circuitos inductivos:

$L\frac{di(t)}{dt} + R_i(t)= - \frac{d\phi(t)}{dt}$ 

donde el flujo total $\Phi(t)$ incluye tanto la contribución del imán como el flujo autoinducido por la corriente $i(t)$ en el anillo.


La fuerza de frenado sobre el imán resulta de la interacción Lorentz entre la corriente inducida $i(t)$ en el anillo y el campo magnético del imán, así, las ecuaciones de movimiento del imán se derivan directamente de la segunda ley de Newton: 

$$m \frac{d^{2}z}{dt^{2}}= mg - F_{z}(t) - F_{\text{disipativa}}$$

donde $m$ es la masa del imán y $F_{\text{disipativa}}$ incluye otras fuerzas de fricción que podrían incluirse o no.

Para estudiar este sistema existen dos aproximaciones posibles y la decisión depende de las facilidades que se encuentren para abordar el problema:

* Partir del imán como un dipolo puntual, ignorando efectos de geometría
* Estudiar el sistema con un imán real, considerando efectos de geometría y magnetización del imán

La primera aproximación, que es puramente analítica, presenta limitaciones significativas en el estudio realista del sistema, pues desprecia efectos cruciales como la geometría finita del imán, el espesor no despreciable del anillo u otros efectos del campo.

Dado que el objetivo del proyecto incluye extender el modelo hacia configuraciones realistas _—que puedan compararse con proyectos experimentales e implementando distintas geometrías y materiales—_ se emplea la librería Magpylib de Python, que nos acerca al segundo método. Esta librería permite modelar imanes de geometría arbitraria y calcular campos magnéticos mediante métodos de integración numérica.

Esta herramienta se utilizará para:
* Recoger los valores de campo y flujo necesarios para la simulación dinámica, evitando el desarrollo de expresiones matemáticas complejas para cada configuración geométrica

**Bibliografía y material de apoyo utilizado:**

[1.] Griffiths, D. J. (2023). Introduction to electrodynamics. Cambridge University Press.
[2.] Purcell, E. M. (2013). Electricity and magnetism. Cambridge university press.
