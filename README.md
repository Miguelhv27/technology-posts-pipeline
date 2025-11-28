# Informe del Proyecto Technology Posts Analytics

Este documento presenta el informe final del proyecto orientado al análisis de publicaciones tecnológicas en redes sociales, desarrollado bajo un enfoque DataOps. A lo largo del trabajo se diseñó, implementó y evaluó un pipeline completo que permitió integrar datos provenientes de Twitter y Reddit, procesarlos mediante técnicas de análisis de texto y estadísticas, y generar métricas confiables acompañadas de visualizaciones interactivas orientadas a la toma de decisiones. Todo el proceso se construyó siguiendo principios de automatización, calidad, versionamiento, observabilidad y gobernanza.

## Descripción del proyecto

El proyecto consistió en la creación de un pipeline automatizado capaz de capturar datos de redes sociales, transformarlos, validar su calidad y analizarlos para identificar tendencias tecnológicas, patrones de engagement y comportamientos relevantes en comunidades digitales. El enfoque estuvo centrado en el dominio de Technology Posts, el cual incluye publicaciones relacionadas con innovación, desarrollo tecnológico y discusión técnica en general. El objetivo fue comprender la relación entre el volumen de menciones y la interacción generada en plataformas complementarias, especialmente Twitter y Reddit.

## Objetivo general

El objetivo principal fue construir un pipeline completamente automatizado que integrara datos estructurados y no estructurados provenientes de redes sociales. Este pipeline debía ser capaz de aplicar técnicas de procesamiento de lenguaje natural, cálculos estadísticos y análisis de series temporales para generar información útil y confiable que pudiera visualizarse mediante dashboards interactivos. Todo el desarrollo debía realizarse siguiendo los principios de DataOps para garantizar trazabilidad, calidad, reproducibilidad y eficiencia.

## Objetivos específicos

El proyecto buscó automatizar la ingesta, transformación y análisis de datos. También fue necesario aplicar técnicas estadísticas como correlaciones y análisis temporal para encontrar relaciones entre variables clave. Se realizó procesamiento de texto para entender el sentimiento general y los temas dominantes en las conversaciones tecnológicas. Además, se construyó un dashboard interactivo para visualizar resultados en tiempo real y se implementaron mecanismos de validación, pruebas y monitoreo para asegurar la calidad y observabilidad del sistema.

## Diseño y arquitectura

La arquitectura del proyecto se construyó considerando las etapas fundamentales de un pipeline de datos: ingesta, validación, transformación, análisis y visualización. Cada etapa fue implementada como un módulo independiente dentro del código, lo que permitió una estructura organizada y escalable. La orquestación se realizó mediante un script que centraliza todas las operaciones y garantiza la ejecución ordenada del flujo completo.





Los componentes principales incluyeron módulos para ingesta de datos desde archivos y fuentes simuladas, validación estructural para prevenir errores tempranos, limpieza y transformación, análisis de texto mediante herramientas de NLP como VADER para calcular sentimiento y LDA para identificar temas, y finalmente el análisis estadístico donde se calcularon correlaciones, se realizaron pruebas de hipótesis y se exploraron tendencias temporales. El dashboard final integró todos estos resultados y permitió navegar entre métricas, filtros y visualizaciones.

## Metodología aplicada
La metodología se desarrolló siguiendo la lógica natural de un pipeline de datos. En la fase de ingesta se incorporaron datos provenientes de Twitter y de un dataset público de Reddit con publicaciones tecnológicas. Se trabajó con datos reales y simulados dependiendo de la disponibilidad y las necesidades del análisis. En la etapa de transformación se realizó limpieza de texto, normalización y creación de nuevas variables que permitieron definir métricas de engagement, duración, sentimiento y temporalidad.

Luego se aplicaron técnicas de procesamiento de lenguaje natural para comprender el tono emocional general presente en las publicaciones. Para ello se utilizó el modelo VADER, que permite asignar un puntaje de sentimiento a cada texto. También se implementó modelado de temas mediante LDA con el fin de identificar categorías recurrentes de conversación dentro del dominio tecnológico.

En la fase de análisis estadístico se calcularon correlaciones entre variables importantes como el volumen de publicaciones y el nivel de engagement registrado en dichas plataformas. Asimismo, se realizaron pruebas ANOVA y t-test para evaluar diferencias entre grupos de datos y se revisaron tendencias mediante análisis de series temporales para identificar patrones estacionales o cambios significativos en los niveles de interacción.

Finalmente, los datos se visualizaron en un dashboard desarrollado en Streamlit que permitió interactuar con las métricas obtenidas. El dashboard incluyó gráficos de evolución temporal, distribuciones de engagement, niveles de sentimiento, temas detectados y correlaciones.

## Resultados principales
El desarrollo del pipeline permitió procesar más de siete mil publicaciones provenientes de Reddit y Twitter. Una vez analizados los datos, se observó que el engagement no se distribuye de manera uniforme. Reddit presentó valores mucho más altos en los niveles máximos de interacción, mientras que Twitter mostró un comportamiento más estable aunque con picos menos pronunciados. El promedio general de engagement fue relativamente bajo, lo cual indica que la mayoría de publicaciones obtiene interacciones pequeñas pero existe un grupo reducido de publicaciones virales que elevan la media significativamente.

En cuanto al sentimiento, los análisis mostraron que el tono general fue ligeramente positivo. El valor promedio del sentimiento calculado fue cercano a cero pero inclinado hacia la positividad. Esto sugiere que, en general, las discusiones tecnológicas tienden a ser neutrales con un leve sesgo positivo. La relación entre el sentimiento y el engagement resultó prácticamente nula. Esto significa que no existe evidencia clara de que las publicaciones con mayor carga emocional generen mayor interacción.

Las correlaciones más relevantes incluyeron una relación moderada positiva entre la actividad de usuarios influyentes y la viralidad de ciertas publicaciones, así como una consistencia interesante en el engagement cruzado entre plataformas. Sin embargo, la relación entre volumen de publicaciones y engagement mostró una correlación leve y negativa, lo que indica que un mayor número de publicaciones no necesariamente se traduce en más interacción.

Los análisis temporales evidenciaron variaciones a lo largo del tiempo que reflejan tendencias en la discusión tecnológica. Se observó que ciertos temas generan picos de interacción en momentos particulares, lo que permitió identificar patrones estacionales y comportamientos relevantes para la toma de decisiones.

## Pruebas automatizadas y registro de logs
Para garantizar la calidad del pipeline se implementaron pruebas automatizadas utilizando pytest. Las pruebas cubrieron estructura y consistencia de datos, operaciones de transformación, cálculos estadísticos y funcionamiento general del pipeline. Además, se integró un flujo de CI/CD mediante GitHub Actions que ejecuta las pruebas, valida la configuración, analiza la calidad del código y revisa aspectos de seguridad en cada ejecución.

El sistema de logs permitió registrar cada ejecución del pipeline de forma completa. Se conservaron marcas de tiempo, mensajes estructurados, identificación de cada etapa del proceso y el estado final de la ejecución. Esto facilitó la detección de errores, el seguimiento del flujo y la recuperación de información necesaria para auditorías y análisis posteriores.

## Principios DataOps aplicados
El desarrollo del proyecto incluyó la aplicación explícita de los principios de DataOps. La automatización estuvo presente en todas las etapas gracias al pipeline end-to-end, diseñado para ejecutarse con un solo comando y mantenido mediante CI/CD. La calidad se abordó mediante validaciones de datos, pruebas unitarias y métricas internas que permitieron asegurar integridad y consistencia. El versionamiento se aplicó tanto al código como a los datos procesados, garantizando reproducibilidad del entorno y de los resultados. La observabilidad se materializó mediante logs, métricas internas y el dashboard final. Finalmente, la gobernanza se mantuvo a través de documentación clara, estructuras organizadas y configuración centralizada.

## Ejecución del proyecto
Para ejecutar el proyecto fue necesario clonar el repositorio correspondiente, instalar las dependencias especificadas, configurar variables de entorno y finalmente ejecutar el pipeline mediante el comando principal. El dashboard se levantó con Streamlit y permitió visualizar los resultados de manera intuitiva. Los tests se ejecutaron de forma independiente para asegurar la validez del pipeline antes de su despliegue.

## Conclusión y mejoras futuras
El proyecto logró construir un pipeline funcional y bien estructurado que permitió analizar publicaciones tecnológicas desde múltiples perspectivas. Se consiguió identificar patrones relevantes de engagement, sentimiento y tendencias, y se generaron correlaciones que aportan al entendimiento del comportamiento social tecnológico. El enfoque DataOps permitió mantener calidad, orden y trazabilidad a lo largo del desarrollo.

Como mejora futura se podría integrar APIs reales para capturar datos en tiempo real y almacenar los resultados en bases de datos especializadas. También sería útil incorporar modelos predictivos que permitan anticipar tendencias tecnológicas y ampliar el análisis a otras plataformas como LinkedIn y TikTok. El uso de datos comerciales reales abriría la posibilidad de correlacionar la actividad social con métricas de negocio, lo que potenciaría aún más el valor del sistema.
