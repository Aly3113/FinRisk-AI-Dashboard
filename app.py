import streamlit as st
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# --- SETĂRI PAGINĂ ---
st.set_page_config(page_title="FinRisk AI Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ FinRisk AI: Supravegherea Riscurilor Financiare")
st.markdown("Proiect de practică: Fraudă, Spălare de Bani și Gestionarea Portofoliilor")
st.divider()

# --- MENIU LATERAL ---
st.sidebar.header("Meniu Navigare")
modul_ales = st.sidebar.radio(
    "Alege modulul de analiză:",
    ("1. Detecție Fraudă", "2. Prevenire Spălare de Bani", "3. Gestionare Portofolii")
)

# ==========================================
# MODULUL 1: DETECȚIE FRAUDĂ
# ==========================================
if modul_ales == "1. Detecție Fraudă":
    st.header("🚨 Modulul de Detecție a Fraudelor (Optimizat cu SMOTE)")
    st.info("Acest modul folosește un algoritm Random Forest și tehnica SMOTE pentru a identifica tranzacțiile financiare nelegitime.")
    
    # --- SETĂRI SIMULARE INTERACTIVE ---
    st.markdown("### ⚙️ Setări Simulare")
    # Adăugăm un slider. 20% din total vor fi folosite pentru testare.
    total_tranzactii = st.slider("Alege numărul total de tranzacții generate (80% Antrenare / 20% Testare):", 
                                 min_value=10000, max_value=50000, step=5000, value=10000)
    
    tranzactii_test = int(total_tranzactii * 0.2)
    st.caption(f"📌 *Din totalul de {total_tranzactii}, modelul va fi testat pe un set de **{tranzactii_test} tranzacții nevăzute**.*")
    
    # Buton interactiv pentru a porni antrenarea
    if st.button("▶️ Rulează Antrenarea Modelului AI", type="primary"):
        
        with st.spinner(f'Generăm {total_tranzactii} de date și antrenăm rețeaua... Te rog așteaptă!'):
            
            # 1. Generare date (folosim variabila total_tranzactii în loc de numărul fix)
            X, y = make_classification(n_samples=total_tranzactii, n_features=10, n_informative=5, 
                                       n_redundant=2, weights=[0.95, 0.05], random_state=42)
            coloane = [f"Feature_{i}" for i in range(1, 11)]
            df = pd.DataFrame(X, columns=coloane)
            df['Este_Frauda'] = y
            
            X_train, X_test, y_train, y_test = train_test_split(df[coloane], df['Este_Frauda'], test_size=0.2, random_state=42)
            
            # 2. Aplicare SMOTE
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            
            # 3. Antrenare model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_smote, y_train_smote)
            
            # 4. Predicții
            predictii = model.predict(X_test)
            
            # Calculăm metricile
            precizie = precision_score(y_test, predictii)
            recall = recall_score(y_test, predictii)
            
        st.success("Modelul a fost antrenat cu succes!")
        
        # --- AFIȘARE REZULTATE VIZUALE ---
        st.subheader("Rezultatele Evaluării")
        
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Total Tranzacții Testate", value=f"{len(y_test)}")
        col2.metric(label="Precizie (Precision)", value=f"{precizie:.2%}", delta="Alarme false reduse")
        col3.metric(label="Rata de Găsire (Recall)", value=f"{recall:.2%}", delta="Procentul de fraude prinse")
        
        st.markdown("---")
        
        # --- MATRICEA DE CONFUZIE MICȘORATĂ ---
        st.subheader("Matricea de Confuzie")
        
        # Împărțim ecranul în două coloane. Punem graficul în prima (ca să fie mai mic) și lăsăm a doua goală
        col_grafic, col_libera = st.columns([1, 1]) 
        
        with col_grafic:
            matrice = confusion_matrix(y_test, predictii)
            
            # Am micșorat figsize de la (6,4) la (4,3)
            fig, ax = plt.subplots(figsize=(4, 3)) 
            sns.heatmap(matrice, annot=True, fmt='d', cmap='Reds', 
                        xticklabels=['Legitim', 'Fraudă'], yticklabels=['Legitim', 'Fraudă'], ax=ax,
                        annot_kws={"size": 10}) # Am făcut textul din căsuțe puțin mai mic
            
            ax.set_ylabel('Adevărul (Realitate)', fontsize=9)
            ax.set_xlabel('Predicția Modelului', fontsize=9)
            ax.tick_params(labelsize=8) # Am micșorat scrisul de pe axe
            
            # use_container_width=True face ca imaginea să nu depășească lățimea coloanei mici
            st.pyplot(fig, use_container_width=True) 

# ==========================================
# MODULUL 2: SPĂLARE DE BANI (AML)
# ==========================================
elif modul_ales == "2. Prevenire Spălare de Bani":
    st.header("🕵️‍♂️ Modulul AML: Analiza Rețelelor Financiare")
    st.info("Acest modul generează o rețea dinamică. Rețeaua suspectă este separată în STÂNGA, iar zgomotul de fond în DREAPTA. Poți folosi checkbox-ul 'Activare Fizică' de sub grafic pentru a opri mișcarea.")
    
    # --- SETĂRI SIMULARE INTERACTIVE ---
    st.markdown("### ⚙️ Setări Simulare Rețea")
    col_setari1, col_setari2 = st.columns(2)
    
    with col_setari1:
        nr_intermediari = st.slider("Număr de conturi intermediare (Smurfs):", min_value=5, max_value=50, value=15)
    with col_setari2:
        nr_normale = st.slider("Număr de tranzacții normale (Zgomot de fond):", min_value=10, max_value=150, value=30)

    if st.button("🔍 Generează și Analizează Rețeaua", type="primary"):
        
        with st.spinner('Construim graful rețelei financiare...'):
            
            net = Network(height='100vh', width='100%', bgcolor='#222222', font_color='white', directed=True)
            
            # === REZOLVAREA ERORII: Setăm Fontul și Fizica împreună! ===
            net.set_options("""
            var options = {
              "nodes": {
                "font": {
                  "size": 25
                }
              },
              "physics": {
                "repulsion": {
                  "centralGravity": 0,
                  "springLength": 300,
                  "springConstant": 0.02,
                  "nodeDistance": 500,
                  "damping": 0.9
                },
                "minVelocity": 0.75,
                "solver": "repulsion"
              }
            }
            """)
            
            nod_sursa = "Cont_Ilicit_Main"
            nod_destinatie = "Cont_Offshore"
            intermediari = [f"Intermediar_{i}" for i in range(1, nr_intermediari + 1)]
            
            # 1. GRUPUL 1: REȚEAUA ILEGALĂ (În STÂNGA)
            net.add_node(nod_sursa, label=nod_sursa, title="Sursă", color='#FFA500', size=40, x=-800, y=-200)
            net.add_node(nod_destinatie, label=nod_destinatie, title="Offshore", color='#FF0000', size=50, x=-800, y=200)
            
            suma_totala_spalata = 0
            for nod in intermediari:
                suma_transfer = random.randint(1000, 9000)
                suma_totala_spalata += suma_transfer
                net.add_node(nod, label=nod, title=f"Transferă: {suma_transfer} RON", color='#87CEFA', size=20, x=random.randint(-700, -300), y=random.randint(-400, 400))
                net.add_edge(nod_sursa, nod, value=suma_transfer/1000, title=f"{suma_transfer} RON")
                net.add_edge(nod, nod_destinatie, value=(suma_transfer-100)/1000, title=f"{suma_transfer - 100} RON")
                
            # 2. GRUPUL 2: ZGOMOTUL DE FOND (În DREAPTA)
            clienti_normali = [f"Client_{i}" for i in range(1, int(nr_normale/2) + 1)]
            magazine = ["Emag", "Supermarket", "Factura_Curent", "Benzinarie", "Chirie"]
            for client in clienti_normali:
                net.add_node(client, label=client, title="Client Normal", color='#90EE90', size=15, x=random.randint(500, 1000), y=random.randint(-500, 500))
            for magazin in magazine:
                net.add_node(magazin, label=magazin, title="Comerciant", color='#D3D3D3', size=25, x=random.randint(500, 1000), y=random.randint(-500, 500))
                
            for _ in range(nr_normale):
                sursa_normala = random.choice(clienti_normali)
                destinatie_normala = random.choice(clienti_normali + magazine)
                if sursa_normala != destinatie_normala:
                    suma_mica = random.randint(50, 500)
                    net.add_edge(sursa_normala, destinatie_normala, value=1, title=f"{suma_mica} RON")
            
            # --- Am șters net.repulsion(...) de aici pentru că e deja configurat mai sus! ---
            
            # Salvare
            nume_fisier = "pyvis_graph.html"
            net.save_graph(nume_fisier)
            
            # Injectăm Javascript personalizat pentru Checkbox Simplificat
            with open(nume_fisier, 'r', encoding='utf-8') as f:
                html_data = f.read()
                
            # === NOU: Javascript corectat pentru oprirea fizicii ===
            custom_js = """
                <div id="physics-container" style="position: absolute; bottom: 10px; left: 10px; z-index: 1000; padding: 5px; background: rgba(0,0,0,0.5); color: white; border: 1px solid #555; border-radius: 5px; font-size: 14px;">
                    <input type="checkbox" id="toggle-physics" checked onchange="network.setOptions({ physics: { enabled: this.checked } });">
                    <label for="toggle-physics"> Activare Fizică</label>
                </div>
            """
            html_data_with_button = html_data.replace("</body>", custom_js + "</body>")
            
            # Afișare
            components.html(html_data_with_button, height=800, scrolling=True)
            
            # --- BUTONUL DE FULLSCREEN / DESCARCARE ---
            st.markdown("---")
            col_concluzie, col_buton = st.columns([2, 1])
            with col_concluzie:
                st.error(f"⚠️ **ALERTĂ DE SISTEM:** Modelul a detectat o rețea de tip 'Smurfing'!")
                st.warning(f"S-a încercat spălarea sumei de **{suma_totala_spalata:,} RON**.")
            with col_buton:
                st.download_button(
                    label="📺 Deschide Fullscreen (Exportă Graf)",
                    data=html_data_with_button, # Exportăm fișierul care conține și butonul de fizică
                    file_name="Analiza_Retea_AML.html",
                    mime="text/html",
                    help="Descarcă fișierul și deschide-l cu dublu-click în browser."
                )
    
# ==========================================
# MODULUL 3: GESTIONARE PORTOFOLII
# ==========================================
elif modul_ales == "3. Gestionare Portofolii":
    st.header("📈 Modulul de Optimizare și Risc Portofoliu")
    st.info("Acest modul extrage date financiare REALE în timp real folosind Yahoo Finance. Analizăm evoluția prețului, trendul (SMA 20) și volatilitatea acțiunilor.")
    
    # --- SETĂRI PORTOFOLIU ---
    st.markdown("### ⚙️ Setări Portofoliu")
    
    col_port1, col_port2 = st.columns(2)
    with col_port1:
        actiuni_alese = st.multiselect(
            "Alege acțiunile pentru analiză (Simboluri Bursiere):",
            ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META"],
            default=["AAPL", "NVDA"]
        )
    with col_port2:
        perioada = st.selectbox(
            "Alege perioada de analiză:",
            ["3mo", "6mo", "1y", "2y", "5y", "max"], # Am scos 1mo pentru ca SMA 20 are nevoie de minim 20 de zile de istoric
            index=2 # Default este '1y'
        )

    if st.button("📊 Descarcă Datele și Analizează", type="primary"):
        if not actiuni_alese:
            st.warning("Te rog să alegi cel puțin o acțiune din listă!")
        else:
            with st.spinner(f'Descărcăm datele live de pe bursă pentru {len(actiuni_alese)} acțiuni...'):
                try:
                    # 1. Descărcăm datele de pe Yahoo Finance
                    date_bursa = yf.download(actiuni_alese, period=perioada)['Close']
                    
                    if isinstance(date_bursa, pd.Series):
                        date_bursa = date_bursa.to_frame(name=actiuni_alese[0])
                        
                    date_bursa = date_bursa.dropna()
                    
                    # === NOU: CALCULĂM MEDIA MOBILĂ (SMA 20) PENTRU FIECARE ACȚIUNE ===
                    for actiune in actiuni_alese:
                        nume_sma = f"{actiune} (SMA 20)"
                        # rolling(window=20).mean() calculează media din ultimele 20 de zile
                        date_bursa[nume_sma] = date_bursa[actiune].rolling(window=20).mean()
                    
                    # 2. Afișăm GRAFICUL INTERACTIV cu Plotly
                    st.subheader("Evoluția Prețului și Indicatorul de Trend (SMA 20)")
                    st.caption("💡 *Linia continuă este prețul real. Linia întreruptă (de aceeași culoare) este Media Mobilă (SMA 20).*")
                    
                    fig = px.line(date_bursa, x=date_bursa.index, y=date_bursa.columns,
                                  labels={'value': 'Preț Închidere (USD)', 'Date': 'Data', 'variable': 'Legendă'},
                                  template="plotly_dark")
                    
                    # === NOU: LOGICA PENTRU CULORI IDENTICE ȘI LINII ÎNTRERUPTE ===
                    
                    # Pasul A: Salvăm culorile acțiunilor "părinte" într-un dicționar (memorie)
                    culori_actiuni = {}
                    for trace in fig.data:
                        if "SMA 20" not in trace.name:
                            culori_actiuni[trace.name] = trace.line.color # Memorăm culoarea (ex: AAPL -> Albastru)
                            trace.line.width = 2.5 # Facem linia reală mai groasă
                            
                    # Pasul B: Modificăm liniile SMA să preia culoarea și să aibă spații mai mari ('dash')
                    for trace in fig.data:
                        if "SMA 20" in trace.name:
                            # Tăiem particula " (SMA 20)" ca să aflăm numele de bază (ex: "AAPL")
                            nume_baza = trace.name.replace(" (SMA 20)", "")
                            
                            # Aplicăm aceeasi culoare ca a prețului real
                            trace.line.color = culori_actiuni[nume_baza] 
                            
                            # 'dash' face liniile întrerupte mai vizibile și distanțate față de 'dot'
                            trace.line.dash = 'dash' 
                            trace.line.width = 1.5
                            
                    fig.update_layout(hovermode="x unified", legend_title_text="Acțiuni & Indicatori") 
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 3. CALCULĂM RISCUL (Volatilitatea) ȘI RANDAMENTUL
                    st.markdown("---")
                    st.subheader("Analiza de Risc și Performanță")
                    
                    # Calculăm pe baza datelor originale (fără coloanele SMA)
                    randamente_zilnice = date_bursa[actiuni_alese].pct_change().dropna()
                    
                    coloane_metrice = st.columns(len(actiuni_alese))
                    
                    for i, actiune in enumerate(actiuni_alese):
                        with coloane_metrice[i]:
                            pret_initial = date_bursa[actiune].iloc[0]
                            pret_final = date_bursa[actiune].iloc[-1]
                            crestere_totala = ((pret_final - pret_initial) / pret_initial) * 100
                            
                            import numpy as np
                            volatilitate = randamente_zilnice[actiune].std() * np.sqrt(252) * 100
                            
                            st.markdown(f"#### {actiune}")
                            st.metric(label="Preț Curent", value=f"${pret_final:.2f}", delta=f"{crestere_totala:.2f}% (Evoluție)")
                            
                            if volatilitate > 40:
                                st.error(f"⚠️ Risc Ridicat: Volatilitate {volatilitate:.1f}%")
                            elif volatilitate > 25:
                                st.warning(f"⚡ Risc Mediu: Volatilitate {volatilitate:.1f}%")
                            else:
                                st.success(f"✅ Risc Scăzut: Volatilitate {volatilitate:.1f}%")

                except Exception as e:
                    st.error(f"A apărut o eroare la descărcarea datelor: {e}")

