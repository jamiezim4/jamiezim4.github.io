---
layout: post
author: Jamie Zimmerman
---

See my GitHub for this project: <a href="https://github.com/jamiezim4/ml-predicting-olympic-marathon">ml-predicting-olympic-marathon</a>.

<h2> Using Machine Learning to Predict the Winner of the Women's Marathon at the Paris 2024 Olympics 🥇🏃‍♀️ </h2>

I'm a big fan and spectator of professional endurance sports, Women's distance running in particular, and I run quite a bit myself, so I decided to combine my programming skills and running fanaticism together to stretch my machine-learning chops. __I wrote this project to predict the winner of the Women's Marathon at the 2024 Paris Olympics.__ I trained a model on data from competitor's past performances: what races they ran, in which countries and when, and most importantly what time they ran. It also includes their birth date and their nationality.

I got training data from World Athletics. I didn't scrape their website HTML, but by inspecting the page I was able to find out that they have a GraphQL API that drives their data retrieval. (They don't know that I've done this, but it's a public site and I'm not DDOS-ing them or making money off them)

I used a regression model (not classification) because I want to predict continuous values - each competitor's race time. Then I made a prediction on each athlete in the new venue on the new date, got their race time, and sorted by time to see who will medal in this event.



#### Model Predictions
Here's the full output of predictions for each athlete's race time:

```
Polynomial Model Accuracy score is:  0.3874858240673147
---------- Athlete Predicted Results ----------
             SCHLUMPF, Fabienne (SUI): 02:52:46
                ELMORE, Malindi (CAN): 02:56:36
           MCCORMACK, Fionnuala (IRL): 03:30:02
                WEIGHTMAN, Lisa (AUS): 03:34:44
                 ALEMU, Megertu (ETH): 03:37:13
                      XIA, Yuyu (CHN): 03:37:53
               VAN ZYL, Irvette (RSA): 03:38:56
                  ZHANG, Deshun (CHN): 03:40:01
                  TESFU, Dolshi (ERI): 03:40:25
                        BAI, Li (CHN): 03:40:37
         SHANKULE, Amane Beriso (ETH): 03:43:12
                  WOLDU, Mekdes (FRA): 03:45:04
                  MAEDA, Honami (JPN): 03:45:34
                 KOSGEI, Brigid (KEN): 03:45:35
                  TIYOURI, Maor (ISR): 03:46:02
                  TAHIRI, Rahma (MAR): 03:47:48
                  BEKELE, Helen (SUI): 03:47:49
                   SUZUKI, Yuka (JPN): 03:48:23
           CHELANGAT, Mercyline (UGA): 03:48:30
              JOHANNES, Helalia (NAM): 03:48:32
                  ICHIYAMA, Mao (JPN): 03:50:45
                     HOSODA, Ai (JPN): 03:50:46
                 LOKEDI, Sharon (KEN): 03:50:51
                ESHETE, Shitaye (BRN): 03:51:14
            GEBRESLASE, Gotytom (ETH): 03:51:17
       CHUMBA, Eunice Chebichii (BRN): 03:51:22
                  CHELIMO, Rose (BRN): 03:51:36
               SAKILU, Jackline (TAN): 03:51:38
             JEPCHIRCHIR, Peres (KEN): 03:51:44
          BAYARTSOGT, Munkhzaya (MGL): 03:52:14
            KEJETA, Melat Yisak (GER): 03:52:42
  MAKATISI, Mokulubete Blandina (LES): 03:52:55
                MAAYOUF, Majida (ESP): 03:53:25
              SHAURI, Magdalena (TAN): 03:53:45
             BORELLI, Florencia (ARG): 03:54:02
                BJELJAC, Bojana (CRO): 03:54:40
                   STEYN, Gerda (RSA): 03:57:04
         MUKANDANGA, Clementine (RWA): 03:58:08
             FARKOUSSI, Kaoutar (MAR): 03:58:24
                  HASSAN, Sifan (NED): 03:59:46
                  OBIRI, Hellen (KEN): 04:01:08
                CHESANG, Stella (UGA): 04:01:13
                   MAYER, Julia (AUT): 04:01:36
              PURDUE, Charlotte (GBR): 04:02:14
           RICHARDSSON, Camilla (FIN): 04:02:28
             CHEPTEGEI, Rebecca (UGA): 04:04:12
                 ORJUELA, Angie (COL): 04:04:21
                O'KEEFFE, Fiona (USA): 04:04:42
               STENSON, Jessica (AUS): 04:05:20
                  DIVER, Sinead (AUS): 04:05:37
        MERINGOR, Delvine Relin (ROU): 04:06:52
                FRENCH, Camille (NZL): 04:07:18
                 TEJEDA, Gladys (PER): 04:08:07
     GALBADRAKH, Khishigsaikhan (MGL): 04:08:30
                 GASHAW, Tigist (BRN): 04:09:34
              CHACHA, Rosa Alva (ECU): 04:09:37
OUHADDOU NAFIE, Fatima Azzahraa (ESP): 04:09:43
                ROJAS, Luz Mery (PER): 04:10:07
                  ASSEFA, Tigst (ETH): 04:10:30
        SALPETER, Lonah Chemtai (ISR): 04:12:54
           PARLOV KOŠTRO, Matea (CRO): 04:13:12
                 SANTOS, Susana (POR): 04:14:04
             TROFIMOVA, Sardana (KGZ): 04:14:46
                  LUIJTEN, Anne (NED): 04:15:12
            MELLY, Joan Chelimo (ROU): 04:15:24
                 PERRIER, Marie (MRI): 04:16:09
            MAMAZHANOVA, Zhanna (KAZ): 04:16:19
                 MACH, Angelika (POL): 04:16:23
             GREGSON, Genevieve (AUS): 04:18:01
                 JULIEN, Mélody (FRA): 04:18:14
              STEWARTOVÁ, Moira (CZE): 04:19:14
                  SISSON, Emily (USA): 04:19:18
               MCCLAIN, Jessica (USA): 04:21:32
           LISOWSKA, Aleksandra (POL): 04:22:11
                 OCAMPO, Daiana (ARG): 04:22:40
            SCHÖNEBORN, Deborah (GER): 04:23:32
                   TRAPP, Manon (FRA): 04:23:55
              HOTTENROTT, Laura (GER): 04:26:44
    HERNANDEZ FLORES, Margarita (MEX): 04:27:53
         HAUGER-THACKERY, Calli (GBR): 04:28:54
               VALDIVIA, Thalia (PER): 04:29:35
                MAYER, Domenika (GER): 04:30:21
               HROCHOVÁ, Tereza (CZE): 04:39:32
               SOLER, Meritxell (ESP): 04:40:06
              VERBRUGGEN, Hanne (BEL): 04:42:52
              NAVARRETE, Esther (ESP): 04:44:07
                 EPIS, Giovanna (ITA): 04:47:31
      CRISTIAN MOSCOTE, Citlali (MEX): 04:48:33
                 HERBIET, Chloé (BEL): 04:49:28
             WIKSTRÖM, Carolina (SWE): 04:50:25
          NYAHORA, Rutendo Joan (ZIM): 04:51:46
                 ROLLIN, Méline (FRA): 04:54:42
              YAREMCHUK, Sofiia (ITA): 04:57:19
 ORTIZ MOROCHO, Silvia Patricia (ECU): 05:03:30
           GRANJA, Mary Zenaida (ECU): 05:03:32
              LINDWURM, Dakotah (USA): 05:06:10
        GARDADI, Fatima Ezzahra (MAR): 05:09:12
                   EVANS, Clara (GBR): 05:11:20
                  OLDKNOW, Cian (RSA): 05:18:38
                   HARVEY, Rose (GBR): 05:42:17
```


