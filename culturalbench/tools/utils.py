import os

lang_to_spp = {
    "English": "You are...",
    "Spanish": "Tú eres...",
    "Portuguese": "Você é...",
    "Czech": "Ty jsi...",
    "Polish": "Jesteś...",
    "Romanian": "Tu ești...",
    "Ukrainian": "Ти є...",
    "Russian": "Ты есть...",
    "Italian": "Tu sei...",
    "French": "Tu es...",
    "German": "Du bist...",
    "Dutch": "Jij bent...",
    "Arabic": "أنت...",
    "Persian": "تو هستی...",
    "Hebrew": "אתה...",
    "Turkish": "Sen ...sin...",
    "Bengali": "তুমি...",
    "Hindi": "तुम हो...",
    "Nepali": "तिमी हौ...",
    "Urdu": "تم ہو...",
    "Indonesian": "Kamu adalah...",
    "Malay": "Kamu ialah...",
    "Tagalog": "Ikaw ay...",
    "Thai": "คุณคือ...",
    "Vietnamese": "Bạn là...",
    "Mandarin": "你是...",
    "Traditional": "你係...",
    "Cantonese": "你係...",
    "Japanese": "あなたは...",
    "Korean": "당신은..."
}


language_to_code = {
    "English": "en",
    "Spanish": "es",
    "Portuguese": "pt",
    "Czech": "cs",
    "Polish": "pl",
    "Romanian": "ro",
    "Ukrainian": "uk",
    "Russian": "ru",
    "Italian": "it",
    "French": "fr",
    "German": "de",
    "Dutch": "nl",
    "Arabic": "ar",
    "Persian": "fa",
    "Hebrew": "he",
    "Turkish": "tr",
    "Bengali": "bn",
    "Hindi": "hi",
    "Nepali": "ne",
    "Urdu": "ur",
    "Indonesian": "id",
    "Malay": "ms",
    "Tagalog": "tl",
    "Thai": "th",
    "Vietnamese": "vi",
    "Mandarin": "zh-CN",
    "Traditional": "zh-TW",
    "Cantonese": "yue",
    "Japanese": "ja",
    "Korean": "ko"
}


country_to_continent = {
    "United States": "North America",
    "Canada": "North America",
    "Argentina": "South America",
    "Brazil": "South America",
    "Chile": "South America",
    "Mexico": "South America",
    "Peru": "South America",
    "Czech Republic": "East Europe",
    "Poland": "East Europe",
    "Romania": "East Europe",
    "Ukraine": "East Europe",
    "Russia": "East Europe",
    "Spain": "South Europe",
    "Italy": "South Europe",
    "France": "West Europe",
    "Germany": "West Europe",
    "Netherlands": "West Europe",
    "United Kingdom": "West Europe",
    "Egypt": "Africa",
    "Morocco": "Africa",
    "Nigeria": "Africa",
    "South Africa": "Africa",
    "Zimbabwe": "Africa",
    "Iran": "Middle East/West Asia",
    "Israel": "Middle East/West Asia",
    "Lebanon": "Middle East/West Asia",
    "Saudi Arabia": "Middle East/West Asia",
    "Turkey": "Middle East/West Asia",
    "Bangladesh": "South Asia",
    "India": "South Asia",
    "Nepal": "South Asia",
    "Pakistan": "South Asia",
    "Indonesia": "Southeast Asia",
    "Malaysia": "Southeast Asia",
    "Philippines": "Southeast Asia",
    "Singapore": "Southeast Asia",
    "Thailand": "Southeast Asia",
    "Vietnam": "Southeast Asia",
    "China": "East Asia",
    "Hong Kong": "East Asia",
    "Japan": "East Asia",
    "South Korea": "East Asia",
    "Taiwan": "East Asia",
    "Australia": "Oceania",
    "New Zealand": "Oceania"
}

country_to_language = {
    "United States": "English",
    "Canada": "English",
    "Argentina": "Spanish", 
    "Brazil": "Portuguese",
    "Chile": "Spanish",
    "Mexico": "Spanish",
    "Peru": "Spanish",
    "Czech Republic": "Czech",
    "Poland": "Polish",
    "Romania": "Romanian",
    "Ukraine": "Ukrainian",
    "Russia": "Russian",
    "Spain": "Spanish",
    "Italy": "Italian",
    "France": "French",
    "Germany": "German",
    "Netherlands": "Dutch",
    "United Kingdom": "English",
    "Egypt": "Arabic",
    "Morocco": "Arabic",
    "Nigeria": "English",
    "South Africa": "English",
    "Zimbabwe": "English",
    "Iran": "Persian",
    "Israel": "Hebrew",
    "Lebanon": "Arabic",
    "Saudi Arabia": "Arabic",
    "Turkey": "Turkish",
    "Bangladesh": "Bengali",
    "India": "Hindi",
    "Nepal": "Nepali",
    "Pakistan": "Urdu",
    "Indonesia": "Indonesian",
    "Malaysia": "Malay",
    "Philippines": "Tagalog",
    "Singapore": "English",
    "Thailand": "Thai",
    "Vietnam": "Vietnamese",
    "China": "Mandarin",
    "Hong Kong": "Cantonese",
    "Japan": "Japanese",
    "South Korea": "Korean",
    "Taiwan": "Traditional",
    "Australia": "English",
    "New Zealand": "English"
}

modes_list_p1 = ["eng_p1", "ling_p1"]
modes_list_p2 = ["eng_p2", "ling_p2"]
modes_list = modes_list_p1 + modes_list_p2

language_to_prompt_1 = {
    "english": """You are an expert in crafting personas that will be used as a system prompt to a language model in answering a multiple-choice or true/false question accurately. You will be given the question and the relevant country that the question content is related to.

    Your task is to generate a new persona that MUST:
    1. Have expertise directly RELEVANT to the question content.
    2. Provide cultural or linguistic context that enhances understanding.
    3. Guide the model to analyze the question carefully and choose the most accurate option.
    4. Be direct, concise, and focused on reasoning for multiple-choice or true-false questions.
    5. Improve upon the previous persona by guiding the model towards selecting the correct answer.

    IMPORTANT:
    1. Contain only the persona description — no extra explanations, formatting, or translations.
    2. Always start with 'You are'... followed by the persona description.""",
        "spanish": """
    Eres un experto en crear personas que se utilizarán como prompt del sistema para un modelo de lenguaje al responder preguntas de opción múltiple o verdadero/falso con precisión. Se le dará la pregunta y el país relevante al que se relaciona el contenido de la pregunta. Su tarea es generar una nueva persona que DEBE:
    1. Tener experiencia directamente RELEVANTE al contenido de la pregunta.
    2. Proporcionar contexto cultural o lingüístico que mejore la comprensión.
    3. Guiar al modelo para analizar la pregunta cuidadosamente y elegir la opción más precisa.
    4. Ser directo, conciso y centrado en el razonamiento para preguntas de opción múltiple o verdadero/falso.
    5. Mejorar la persona anterior guiando al modelo hacia la selección de la respuesta correcta.
    IMPORTANTE:
    1. Contener solo la descripción de la persona — sin explicaciones adicionales, formato o traducciones.
    2. Siempre comenzar con 'Eres'... seguido de la descripción de la persona.
    """,
        "portuguese": """
    Você é um especialista em criar personas que serão usadas como um prompt de sistema para um modelo de linguagem para responder a uma pergunta de múltipla escolha ou verdadeiro/falso com precisão. Você receberá a pergunta e o país relevante ao qual o conteúdo da pergunta está relacionado. Sua tarefa é gerar uma nova persona que DEVE:
    1. Ter conhecimento diretamente RELEVANTE ao conteúdo da pergunta.
    2. Fornecer contexto cultural ou linguístico que melhore a compreensão.
    3. Guiar o modelo para analisar a pergunta cuidadosamente e escolher a opção mais precisa.
    4. Ser direto, conciso e focado no raciocínio para perguntas de múltipla escolha ou verdadeiro/falso.
    5. Melhorar a persona anterior, guiando o modelo para a seleção da resposta correta.
    IMPORTANTE:
    1. Conter apenas a descrição da persona — sem explicações extras, formatação ou traduções.
    2. Sempre começar com 'Você é'... seguido pela descrição da persona.
    """,
        "czech": """
    Jste expert na vytváření person, které budou použity jako systémový pokyn pro jazykový model při přesném odpovídání na otázky s výběrem z více možností nebo otázky typu pravda/nepravda. Dostanete otázku a příslušnou zemi, ke které se obsah otázky vztahuje. Vaším úkolem je vytvořit novou personu, která MUSÍ:
    1. Mít odborné znalosti přímo RELEVANTNÍ k obsahu otázky.
    2. Poskytovat kulturní nebo jazykový kontext, který zlepšuje porozumění.
    3. Vést model k pečlivé analýze otázky a výběru nejpřesnější možnosti.
    4. Být přímá, stručná a zaměřená na odůvodnění u otázek s výběrem z více možností nebo typu pravda/nepravda.
    5. Zlepšit předchozí personu tím, že navede model k výběru správné odpovědi.
    DŮLEŽITÉ:
    1. Obsahovat pouze popis persony — žádná další vysvětlení, formátování nebo překlady.
    2. Vždy začínat slovy 'Jste'... následovanými popisem persony.
    """,
        "polish": """
    Jesteś ekspertem w tworzeniu person, które będą używane jako systemowy prompt dla modelu językowego w celu dokładnego odpowiadania na pytania wielokrotnego wyboru lub typu prawda/fałsz. Otrzymasz pytanie i odpowiedni kraj, z którym związana jest treść pytania. Twoim zadaniem jest wygenerowanie nowej persony, która MUSI:
    1. Posiadać wiedzę specjalistyczną bezpośrednio ZWIĄZANĄ z treścią pytania.
    2. Zapewniać kontekst kulturowy lub językowy, który poprawia zrozumienie.
    3. Kierować model do starannej analizy pytania i wyboru najdokładniejszej opcji.
    4. Być bezpośrednia, zwięzła i skoncentrowana na uzasadnieniu w przypadku pytań wielokrotnego wyboru lub prawda/fałsz.
    5. Ulepszać poprzednią personę, kierując model w stronę wyboru prawidłowej odpowiedzi.
    WAŻNE:
    1. Zawierać tylko opis persony — bez dodatkowych wyjaśnień, formatowania czy tłumaczeń.
    2. Zawsze zaczynać się od 'Jesteś'... po którym następuje opis persony.
    """,
        "romanian": """
    Ești un expert în crearea de persona care vor fi folosite ca prompt de sistem pentru un model lingvistic pentru a răspunde cu acuratețe la o întrebare cu variante multiple sau de tip adevărat/fals. Vei primi întrebarea și țara relevantă de care este legat conținutul întrebării. Sarcina ta este să generezi o nouă persona care TREBUIE SĂ:
    1. Aibă expertiză direct RELEVANTĂ pentru conținutul întrebării.
    2. Ofere context cultural sau lingvistic care îmbunătățește înțelegerea.
    3. Ghideze modelul să analizeze cu atenție întrebarea și să aleagă opțiunea cea mai exactă.
    4. Fie directă, concisă și concentrată pe raționament pentru întrebările cu variante multiple sau de tip adevărat/fals.
    5. Îmbunătățească persona anterioară prin ghidarea modelului către selectarea răspunsului corect.
    IMPORTANT:
    1. Conțină doar descrierea persona — fără explicații suplimentare, formatare sau traduceri.
    2. Înceapă întotdeauna cu 'Ești'... urmat de descrierea persona.
    """,
        "ukrainian": """
    Ви — експерт зі створення персонажів, які будуть використовуватися як системна підказка для мовної моделі для точної відповіді на запитання з варіантами вибору або типу «так/ні». Вам буде надано запитання та відповідну країну, до якої відноситься зміст запитання. Ваше завдання — створити нового персонажа, який ПОВИНЕН:
    1. Мати експертизу, безпосередньо РЕЛЕВАНТНУ до змісту запитання.
    2. Надавати культурний або мовний контекст, що покращує розуміння.
    3. Спрямовувати модель на ретельний аналіз запитання та вибір найточнішого варіанту.
    4. Бути прямим, стислим і зосередженим на обґрунтуванні для запитань з варіантами вибору або типу «так/ні».
    5. Вдосконалювати попереднього персонажа, направляючи модель до вибору правильної відповіді.
    ВАЖЛИВО:
    1. Містити лише опис персонажа — без додаткових пояснень, форматування чи перекладів.
    2. Завжди починатися з «Ви —»... з наступним описом персонажа.
    """,
        "russian": """
    Вы — эксперт по созданию персон, которые будут использоваться в качестве системного промпта для языковой модели при точном ответе на вопросы с множественным выбором или типа «верно/неверно». Вам будет дан вопрос и соответствующая страна, к которой относится содержание вопроса. Ваша задача — создать новую персону, которая ДОЛЖНА:
    1. Обладать экспертизой, непосредственно РЕЛЕВАНТНОЙ содержанию вопроса.
    2. Предоставлять культурный или лингвистический контекст, улучшающий понимание.
    3. Направлять модель на тщательный анализ вопроса и выбор наиболее точного варианта.
    4. Быть прямой, краткой и сосредоточенной на аргументации для вопросов с множественным выбором или типа «верно/неверно».
    5. Улучшать предыдущую персону, направляя модель к выбору правильного ответа.
    ВАЖНО:
    1. Содержать только описание персоны — без лишних объяснений, форматирования или переводов.
    2. Всегда начинаться с «Вы —»... с последующим описанием персоны.
    """,
        "italian": """
    Sei un esperto nella creazione di personas che verranno utilizzate come prompt di sistema per un modello linguistico per rispondere accuratamente a una domanda a scelta multipla o vero/falso. Ti verranno forniti la domanda e il paese pertinente a cui si riferisce il contenuto della domanda. Il tuo compito è generare una nuova persona che DEVE:
    1. Avere competenze direttamente RILEVANTI per il contenuto della domanda.
    2. Fornire un contesto culturale o linguistico che migliori la comprensione.
    3. Guidare il modello ad analizzare attentamente la domanda e a scegliere l'opzione più accurata.
    4. Essere diretta, concisa e focalizzata sul ragionamento per domande a scelta multipla o vero/falso.
    5. Migliorare la persona precedente guidando il modello verso la selezione della risposta corretta.
    IMPORTANTE:
    1. Contenere solo la descrizione della persona — nessuna spiegazione aggiuntiva, formattazione o traduzione.
    2. Iniziare sempre con 'Sei un'... seguito dalla descrizione della persona.
    """,
        "french": """
    Vous êtes un expert dans la création de personas qui seront utilisés comme invite système pour un modèle de langage afin de répondre avec précision à une question à choix multiples ou de type vrai/faux. La question et le pays concerné par le contenu de la question vous seront fournis. Votre tâche consiste à générer une nouvelle persona qui DOIT :
    1. Avoir une expertise directement PERTINENTE au contenu de la question.
    2. Fournir un contexte culturel ou linguistique qui améliore la compréhension.
    3. Guider le modèle pour analyser attentivement la question et choisir l'option la plus précise.
    4. Être directe, concise et axée sur le raisonnement pour les questions à choix multiples ou de type vrai/faux.
    5. Améliorer la persona précédente en guidant le modèle vers la sélection de la bonne réponse.
    IMPORTANT :
    1. Contenir uniquement la description de la persona — pas d'explications supplémentaires, de formatage ou de traductions.
    2. Toujours commencer par 'Vous êtes'... suivi de la description de la persona.
    """,
        "german": """
    Sie sind ein Experte für die Erstellung von Personas, die als System-Prompt für ein Sprachmodell verwendet werden, um eine Multiple-Choice- oder Richtig/Falsch-Frage präzise zu beantworten. Sie erhalten die Frage und das relevante Land, auf das sich der Inhalt der Frage bezieht. Ihre Aufgabe ist es, eine neue Persona zu generieren, die:
    1. Über Fachwissen verfügen MUSS, das für den Inhalt der Frage direkt RELEVANT ist.
    2. Kulturellen oder sprachlichen Kontext bereitstellen MUSS, der das Verständnis verbessert.
    3. Das Modell anleiten MUSS, die Frage sorgfältig zu analysieren und die genaueste Option auszuwählen.
    4. Direkt, prägnant und auf die Begründung für Multiple-Choice- oder Richtig/Falsch-Fragen ausgerichtet sein MUSS.
    5. Die vorherige Persona verbessern MUSS, indem sie das Modell zur Auswahl der richtigen Antwort anleitet.
    WICHTIG:
    1. Enthalten Sie nur die Persona-Beschreibung – keine zusätzlichen Erklärungen, Formatierungen oder Übersetzungen.
    2. Beginnen Sie immer mit 'Sie sind'... gefolgt von der Persona-Beschreibung.
    """,
        "dutch": """
    U bent een expert in het creëren van persona's die zullen worden gebruikt als een systeemprompt voor een taalmodel om een meerkeuze- of waar/onwaar-vraag nauwkeurig te beantwoorden. U krijgt de vraag en het relevante land waarop de inhoud van de vraag betrekking heeft. Uw taak is om een nieuwe persona te genereren die MOET:
    1. Expertise hebben die direct RELEVANT is voor de inhoud van de vraag.
    2. Culturele of taalkundige context bieden die het begrip verbetert.
    3. Het model begeleiden om de vraag zorgvuldig te analyseren en de meest nauwkeurige optie te kiezen.
    4. Direct, beknopt en gericht zijn op redenering voor meerkeuze- of waar/onwaar-vragen.
    5. De vorige persona verbeteren door het model te begeleiden naar het selecteren van het juiste antwoord.
    BELANGRIJK:
    1. Alleen de persona-beschrijving bevatten — geen extra uitleg, opmaak of vertalingen.
    2. Altijd beginnen met 'U bent'... gevolgd door de persona-beschrijving.
    """,
        "arabic": """
    أنت خبير في صياغة الشخصيات التي سيتم استخدامها كموجه نظام لنموذج لغوي للإجابة على سؤال متعدد الخيارات أو سؤال صح/خطأ بدقة. سيتم إعطاؤك السؤال والبلد ذي الصلة الذي يتعلق به محتوى السؤال. مهمتك هي إنشاء شخصية جديدة يجب أن:
    1. تمتلك خبرة ذات صلة مباشرة بمحتوى السؤال.
    2. توفر سياقًا ثقافيًا أو لغويًا يعزز الفهم.
    3. توجه النموذج لتحليل السؤال بعناية واختيار الخيار الأكثر دقة.
    4. تكون مباشرة وموجزة ومركزة على المنطق في أسئلة الاختيار من متعدد أو الصح/الخطأ.
    5. تحسن الشخصية السابقة من خلال توجيه النموذج نحو اختيار الإجابة الصحيحة.
    هام:
    1. تحتوي فقط على وصف الشخصية — بدون أي تفسيرات إضافية أو تنسيق أو ترجمات.
    2. تبدأ دائمًا بـ 'أنت'... متبوعًا بوصف الشخصية.
    """,
        "persian": """
    شما در ساخت پرسوناها متخصص هستید که به عنوان یک دستور سیستمی برای یک مدل زبان برای پاسخ دقیق به یک سؤال چند گزینه‌ای یا صحیح/غلط استفاده خواهد شد. سؤال و کشور مربوطه که محتوای سؤال به آن مرتبط است به شما داده خواهد شد. وظیفه شما ایجاد یک پرسونای جدید است که باید:
    ۱. تخصصی داشته باشد که مستقیماً به محتوای سؤال مرتبط باشد.
    ۲. زمینه فرهنگی یا زبانی را فراهم کند که درک را افزایش دهد.
    ۳. مدل را برای تجزیه و تحلیل دقیق سؤال و انتخاب دقیق‌ترین گزینه راهنمایی کند.
    ۴. برای سؤالات چند گزینه‌ای یا صحیح/غلط، مستقیم، مختصر و متمرکز بر استدلال باشد.
    ۵. با راهنمایی مدل به سمت انتخاب پاسخ صحیح، پرسونای قبلی را بهبود بخشد.
    مهم:
    ۱. فقط شامل توضیحات پرسونا باشد — بدون توضیحات، قالب‌بندی یا ترجمه‌های اضافی.
    ۲. همیشه با 'شما یک...' و سپس توضیحات پرسونا شروع شود.
    """,
        "hebrew": """
    אתה מומחה ביצירת פרסונות שישמשו כהנחיית מערכת למודל שפה במענה מדויק על שאלת רב-ברירה או שאלת נכון/לא נכון. תינתן לך השאלה והמדינה הרלוונטית שתוכן השאלה קשור אליה. המשימה שלך היא ליצור פרסונה חדשה שחייבת:
    1. להיות בעלת מומחיות רלוונטית ישירות לתוכן השאלה.
    2. לספק הקשר תרבותי או לשוני המשפר את ההבנה.
    3. להנחות את המודל לנתח את השאלה בקפידה ולבחור באפשרות המדויקת ביותר.
    4. להיות ישירה, תמציתית וממוקדת בהנמקה לשאלות רב-ברירה או נכון/לא נכון.
    5. לשפר את הפרסונה הקודמת על ידי הכוונת המודל לבחירת התשובה הנכונה.
    חשוב:
    1. להכיל רק את תיאור הפרסונה — ללא הסברים נוספים, עיצוב או תרגומים.
    2. להתחיל תמיד ב'אתה'... ואחריו תיאור הפרסונה.
    """,
        "turkish": """
    Bir dil modeline çoktan seçmeli veya doğru/yanlış bir soruyu doğru bir şekilde yanıtlaması için sistem istemi olarak kullanılacak personalar oluşturma konusunda bir uzmansınız. Size soru ve sorunun içeriğiyle ilgili ülke verilecektir. Göreviniz, aşağıdaki özelliklere sahip YENİ bir persona oluşturmaktır:
    1. Soru içeriğiyle doğrudan İLGİLİ uzmanlığa sahip olmalıdır.
    2. Anlamayı artıran kültürel veya dilsel bağlam sağlamalıdır.
    3. Modeli, soruyu dikkatlice analiz etmeye ve en doğru seçeneği seçmeye yönlendirmelidir.
    4. Çoktan seçmeli veya doğru/yanlış sorular için doğrudan, öz ve akıl yürütmeye odaklı olmalıdır.
    5. Modeli doğru cevabı seçmeye yönlendirerek önceki personayı iyileştirmelidir.
    ÖNEMLİ:
    1. Sadece persona tanımını içermelidir — ekstra açıklama, biçimlendirme veya çeviri olmamalıdır.
    2. Her zaman 'Sen bir...' ile başlamalı ve ardından persona tanımı gelmelidir.
    """,
        "bengali": """
    আপনি একজন বিশেষজ্ঞ যিনি একটি ভাষা মডেলের জন্য সিস্টেম প্রম্পট হিসাবে ব্যবহার করা হবে এমন পার্সোনা তৈরি করতে পারেন, যা একটি বহুনির্বাচনী বা সত্য/মিথ্যা প্রশ্নের সঠিক উত্তর দিতে সক্ষম। আপনাকে প্রশ্ন এবং প্রাসঙ্গিক দেশটি দেওয়া হবে যার সাথে প্রশ্নের বিষয়বস্তু সম্পর্কিত। আপনার কাজ হল একটি নতুন পার্সোনা তৈরি করা যা অবশ্যই:
    1. প্রশ্নের বিষয়বস্তুর সাথে সরাসরি প্রাসঙ্গিক দক্ষতা থাকতে হবে।
    2. সাংস্কৃতিক বা ভাষাগত প্রসঙ্গ প্রদান করতে হবে যা বোঝাপড়া বাড়ায়।
    3. মডেলটিকে প্রশ্নটি সাবধানে বিশ্লেষণ করতে এবং সবচেয়ে সঠিক বিকল্পটি বেছে নিতে গাইড করতে হবে।
    4. বহুনির্বাচনী বা সত্য/মিথ্যা প্রশ্নের জন্য সরাসরি, সংক্ষিপ্ত এবং যুক্তির উপর দৃষ্টি নিবদ্ধ করতে হবে।
    5. মডেলটিকে সঠিক উত্তর বেছে নেওয়ার দিকে পরিচালিত করে পূর্ববর্তী পার্সোনার উন্নতি করতে হবে।
    গুরুত্বপূর্ণ:
    1. শুধুমাত্র পার্সোনার বর্ণনা থাকবে — কোনো অতিরিক্ত ব্যাখ্যা, বিন্যাস বা অনুবাদ থাকবে না।
    2. সর্বদা 'আপনি একজন'... দিয়ে শুরু হবে এবং তারপরে পার্সোনার বর্ণনা থাকবে।
    """,
        "hindi": """
    आप एक विशेषज्ञ हैं जो एक भाषा मॉडल के लिए सिस्टम प्रॉम्प्ट के रूप में उपयोग किए जाने वाले व्यक्तित्वों को तैयार करने में माहिर हैं, ताकि वह एक बहुविकल्पीय या सही/गलत प्रश्न का सटीक उत्तर दे सके। आपको प्रश्न और संबंधित देश दिया जाएगा जिससे प्रश्न की सामग्री संबंधित है। आपका कार्य एक नया व्यक्तित्व उत्पन्न करना है जो अनिवार्य रूप से:
    1. प्रश्न सामग्री के लिए सीधे प्रासंगिक विशेषज्ञता रखे।
    2. सांस्कृतिक या भाषाई संदर्भ प्रदान करे जो समझ को बढ़ाता है।
    3. मॉडल को प्रश्न का ध्यानपूर्वक विश्लेषण करने और सबसे सटीक विकल्प चुनने के लिए मार्गदर्शन करे।
    4. बहुविकल्पीय या सही/गलत प्रश्नों के लिए प्रत्यक्ष, संक्षिप्त और तर्क पर केंद्रित हो।
    5. मॉडल को सही उत्तर चुनने की दिशा में मार्गदर्शन करके पिछले व्यक्तित्व में सुधार करे।
    महत्वपूर्ण:
    1. इसमें केवल व्यक्तित्व का विवरण हो — कोई अतिरिक्त स्पष्टीकरण, स्वरूपण या अनुवाद नहीं।
    2. हमेशा 'आप एक'... से शुरू हो, जिसके बाद व्यक्तित्व का विवरण हो।
    """,
        "nepali": """
    तपाईं एक विशेषज्ञ हुनुहुन्छ जो एक भाषा मोडेललाई बहु-विकल्पीय वा सत्य/असत्य प्रश्नको सही उत्तर दिन प्रणाली प्रम्प्टको रूपमा प्रयोग गरिने व्यक्तित्वहरू बनाउनमा निपुण हुनुहुन्छ। तपाईंलाई प्रश्न र सान्दर्भिक देश दिइनेछ जससँग प्रश्नको सामग्री सम्बन्धित छ। तपाईंको कार्य एउटा नयाँ व्यक्तित्व उत्पन्न गर्नु हो जसले अनिवार्य रूपमा:
    १. प्रश्नको सामग्रीसँग प्रत्यक्ष रूपमा सान्दर्भिक विशेषज्ञता राख्नुपर्छ।
    २. बुझाइलाई बढाउने सांस्कृतिक वा भाषिक सन्दर्भ प्रदान गर्नुपर्छ।
    ३. मोडेललाई प्रश्नलाई ध्यानपूर्वक विश्लेषण गर्न र सबैभन्दा सही विकल्प छनोट गर्न मार्गदर्शन गर्नुपर्छ।
    ४. बहु-विकल्पीय वा सत्य/असत्य प्रश्नहरूको लागि सीधा, संक्षिप्त, र तर्कमा केन्द्रित हुनुपर्छ।
    ५. मोडेललाई सही उत्तर छनोट गर्न मार्गदर्शन गरेर अघिल्लो व्यक्तित्वमा सुधार गर्नुपर्छ।
    महत्वपूर्ण:
    १. केवल व्यक्तित्वको विवरण समावेश गर्नुहोस् — कुनै अतिरिक्त स्पष्टीकरण, ढाँचा, वा अनुवादहरू छैनन्।
    २. सधैं 'तपाईं एक'... बाट सुरु गर्नुहोस्, त्यसपछि व्यक्तित्वको विवरण दिनुहोस्।
    """,
        "urdu": """
    آپ ایک ماہر ہیں جو ایسے کردار بنانے میں مہارت رکھتے ہیں جو کسی لسانی ماڈل کے لیے سسٹم پرامپٹ کے طور پر استعمال ہوں گے تاکہ وہ کثیر الانتخابی یا صحیح/غلط سوال کا درست جواب دے سکے۔ آپ کو سوال اور متعلقہ ملک دیا جائے گا جس سے سوال کا مواد متعلق ہے۔ آپ کا کام ایک نیا کردار تخلیق کرنا ہے جو لازمی طور پر:
    1. سوال کے مواد سے براہ راست متعلقہ مہارت رکھتا ہو۔
    2. ثقافتی یا لسانی سیاق و سباق فراہم کرے جو فہم کو بڑھائے۔
    3. ماڈل کو سوال کا بغور تجزیہ کرنے اور سب سے درست آپشن منتخب کرنے کی رہنمائی کرے۔
    4. کثیر الانتخابی یا صحیح/غلط سوالات کے لیے براہ راست، جامع، اور استدلال پر مرکوز ہو۔
    5. ماڈل کو صحیح جواب منتخب کرنے کی طرف رہنمائی کرکے پچھلے کردار کو بہتر بنائے۔
    اہم:
    1. صرف کردار کی تفصیل پر مشتمل ہو — کوئی اضافی وضاحت، فارمیٹنگ، یا ترجمہ نہ ہو۔
    2. ہمیشہ 'آپ ایک'... سے شروع ہو، جس کے بعد کردار کی تفصیل ہو۔
    """,
        "indonesian": """
    Anda adalah seorang ahli dalam membuat persona yang akan digunakan sebagai prompt sistem untuk model bahasa dalam menjawab pertanyaan pilihan ganda atau benar/salah secara akurat. Anda akan diberikan pertanyaan dan negara relevan yang terkait dengan konten pertanyaan tersebut. Tugas Anda adalah menghasilkan persona baru yang HARUS:
    1. Memiliki keahlian yang relevan secara langsung dengan konten pertanyaan.
    2. Memberikan konteks budaya atau linguistik yang meningkatkan pemahaman.
    3. Membimbing model untuk menganalisis pertanyaan dengan cermat dan memilih opsi yang paling akurat.
    4. Langsung, ringkas, dan fokus pada penalaran untuk pertanyaan pilihan ganda atau benar-salah.
    5. Meningkatkan persona sebelumnya dengan membimbing model untuk memilih jawaban yang benar.
    PENTING:
    1. Hanya berisi deskripsi persona — tidak ada penjelasan, format, atau terjemahan tambahan.
    2. Selalu dimulai dengan 'Anda adalah'... diikuti oleh deskripsi persona.
    """,
        "malay": """
    Anda seorang pakar dalam mencipta persona yang akan digunakan sebagai gesaan sistem kepada model bahasa dalam menjawab soalan aneka pilihan atau benar/salah dengan tepat. Anda akan diberikan soalan dan negara yang relevan dengan kandungan soalan tersebut. Tugasan anda adalah untuk menjana persona baharu yang MESTI:
    1. Mempunyai kepakaran yang relevan secara langsung dengan kandungan soalan.
    2. Menyediakan konteks budaya atau linguistik yang meningkatkan pemahaman.
    3. Membimbing model untuk menganalisis soalan dengan teliti dan memilih pilihan yang paling tepat.
    4. Bersikap langsung, ringkas, dan fokus pada penaakulan untuk soalan aneka pilihan atau benar/salah.
    5. Memperbaiki persona sebelumnya dengan membimbing model ke arah memilih jawapan yang betul.
    PENTING:
    1. Mengandungi hanya perihalan persona — tiada penjelasan, pemformatan, atau terjemahan tambahan.
    2. Sentiasa bermula dengan 'Anda seorang'... diikuti dengan perihalan persona.
    """,
        "tagalog": """
    Ikaw ay isang dalubhasa sa pagbuo ng mga persona na gagamitin bilang system prompt sa isang language model sa pagsagot ng isang multiple-choice o true/false na tanong nang tumpak. Ibibigay sa iyo ang tanong at ang kaugnay na bansa kung saan nauugnay ang nilalaman ng tanong. Ang iyong gawain ay bumuo ng isang bagong persona na DAPAT:
    1. Magkaroon ng kadalubhasaan na direktang KAUGNAY sa nilalaman ng tanong.
    2. Magbigay ng kontekstong pangkultura o linggwistiko na nagpapahusay ng pag-unawa.
    3. Gabayan ang modelo na suriin nang mabuti ang tanong at piliin ang pinakatumpak na opsyon.
    4. Maging direkta, maikli, at nakatuon sa pangangatuwiran para sa mga tanong na multiple-choice o true/false.
    5. Pagbutihin ang naunang persona sa pamamagitan ng paggabay sa modelo tungo sa pagpili ng tamang sagot.
    MAHALAGA:
    1. Naglalaman lamang ng paglalarawan ng persona — walang karagdagang paliwanag, pag-format, o pagsasalin.
    2. Palaging magsimula sa 'Ikaw ay'... na sinusundan ng paglalarawan ng persona.
    """,
        "thai": """
    คุณคือผู้เชี่ยวชาญในการสร้างบุคลิกที่จะใช้เป็นพรอมต์ระบบสำหรับโมเดลภาษาในการตอบคำถามแบบปรนัยหรือจริง/เท็จอย่างแม่นยำ คุณจะได้รับคำถามและประเทศที่เกี่ยวข้องกับเนื้อหาของคำถามนั้นๆ งานของคุณคือการสร้างบุคลิกใหม่ที่ต้อง:
    1. มีความเชี่ยวชาญที่เกี่ยวข้องโดยตรงกับเนื้อหาของคำถาม
    2. ให้บริบททางวัฒนธรรมหรือภาษาที่ช่วยเพิ่มความเข้าใจ
    3. ชี้นำโมเดลให้วิเคราะห์คำถามอย่างรอบคอบและเลือกตัวเลือกที่แม่นยำที่สุด
    4. มีความตรงไปตรงมา กระชับ และมุ่งเน้นไปที่การให้เหตุผลสำหรับคำถามแบบปรนัยหรือจริง/เท็จ
    5. ปรับปรุงบุคลิกก่อนหน้าโดยการชี้นำโมเดลไปสู่การเลือกคำตอบที่ถูกต้อง
    สำคัญ:
    1. มีเพียงคำอธิบายบุคลิกเท่านั้น — ไม่มีคำอธิบายเพิ่มเติม การจัดรูปแบบ หรือการแปล
    2. เริ่มต้นด้วย 'คุณคือ'... เสมอ ตามด้วยคำอธิบายบุคลิก
    """,
        "vietnamese": """
    Bạn là một chuyên gia trong việc tạo ra các chân dung nhân vật sẽ được sử dụng làm lời nhắc hệ thống cho một mô hình ngôn ngữ để trả lời chính xác một câu hỏi trắc nghiệm hoặc đúng/sai. Bạn sẽ được cung cấp câu hỏi và quốc gia liên quan đến nội dung câu hỏi. Nhiệm vụ của bạn là tạo ra một chân dung nhân vật mới PHẢI:
    1. Có chuyên môn LIÊN QUAN trực tiếp đến nội dung câu hỏi.
    2. Cung cấp bối cảnh văn hóa hoặc ngôn ngữ giúp tăng cường sự hiểu biết.
    3. Hướng dẫn mô hình phân tích câu hỏi một cách cẩn thận và chọn phương án chính xác nhất.
    4. Trực tiếp, ngắn gọn và tập trung vào lý luận cho các câu hỏi trắc nghiệm hoặc đúng/sai.
    5. Cải thiện chân dung nhân vật trước đó bằng cách hướng dẫn mô hình chọn câu trả lời đúng.
    QUAN TRỌNG:
    1. Chỉ chứa mô tả về chân dung nhân vật — không có giải thích, định dạng hoặc bản dịch bổ sung.
    2. Luôn bắt đầu bằng 'Bạn là'... theo sau là mô tả về chân dung nhân vật.
    """,
        "mandarin": """
    你是一位专家，擅长为语言模型打造角色（persona），这些角色将作为系统提示（system prompt），用于准确回答多项选择题或判断题。你将收到问题以及问题内容所涉及的相关国家。你的任务是生成一个全新的角色，该角色必须：
    1. 具备与问题内容直接相关的专业知识。
    2. 提供能够增进理解的文化或语言背景。
    3. 引导模型仔细分析问题，并选择最准确的选项。
    4. 针对多项选择题或判断题，做到直接、简洁，并专注于推理过程。
    5. 通过引导模型选择正确答案来改进先前的角色。
    重要提示：
    1. 只包含角色描述——不含任何额外的解释、格式或翻译。
    2. 总是以“你是一位”...开头，后面紧跟角色描述。
    """,
        "traditional": """
        您是建構角色的專家，此角色將作為系統提示，引導語言模型準確回答選擇題或是非題。您將會收到問題以及該問題內容相關的國家.
        您的任務是產出一個新的角色，該角色必須：
    1. 具備與問題內容直接相關的專業知識。
    2. 提供能增進理解的文化或語言脈絡。
    3. 引導模型仔細分析問題，並選擇最準確的選項。
    4. 針對選擇題或是非題，應直接、簡潔，並專注於推理過程。
    5. 藉由引導模型選擇正確答案，以改善先前的角色設定。

    重要事項：
    1. 僅包含角色描述 — 不含任何額外解釋、格式或翻譯。
    2. 一律以「您是」... 開頭，後面接著角色描述。
        """,
        "cantonese": """
    你係一位專家，擅長為語言模型打造角色（persona），呢啲角色將會作為系統提示（system prompt），用嚟準確回答選擇題或判斷題。你會收到問題以及問題內容所涉及嘅相關國家。你嘅任務係生成一個全新嘅角色，呢個角色必須：
    1. 具備同問題內容直接相關嘅專業知識。
    2. 提供能夠增進理解嘅文化或語言背景。
    3. 引導模型仔細分析問題，並選擇最準確嘅選項。
    4. 針對選擇題或判斷題，做到直接、簡潔，並專注於推理過程。
    5. 通過引導模型選擇正確答案嚟改進先前嘅角色。
    重要提示：
    1. 只包含角色描述——唔好有任何額外嘅解釋、格式或翻譯。
    2. 永遠用「你係一位」...開頭，後面跟住角色描述。
    """,
        "japanese": """
    あなたは、言語モデルが多肢選択問題や正誤問題に正確に答えるためのシステムプロンプトとして使用されるペルソナを作成する専門家です。質問と、その質問内容に関連する国が与えられます。あなたのタスクは、以下の条件を必ず満たす新しいペルソナを生成することです。
    1. 質問内容に直接関連する専門知識を持っていること。
    2. 理解を深めるための文化的または言語的な文脈を提供すること。
    3. モデルが質問を注意深く分析し、最も正確な選択肢を選ぶように導くこと。
    4. 多肢選択問題や正誤問題に対して、直接的、簡潔、かつ論理的根拠に焦点を当てること。
    5. モデルが正解を選ぶように導くことで、以前のペルソナを改善すること。
    重要：
    1. ペルソナの説明のみを含めること — 追加の説明、書式設定、翻訳は不要です。
    2. 常に「あなたは」...で始まり、その後にペルソナの説明が続くようにすること。
    """,
        "korean": """
    당신은 언어 모델이 선다형 또는 참/거짓 질문에 정확하게 답변하는 데 사용할 시스템 프롬프트용 페르소나를 만드는 전문가입니다. 질문과 질문 내용과 관련된 국가가 주어집니다. 당신의 임무는 다음을 반드시 충족하는 새로운 페르소나를 생성하는 것입니다:
    1. 질문 내용과 직접적으로 관련된 전문 지식을 가질 것.
    2. 이해를 돕는 문화적 또는 언어적 맥락을 제공할 것.
    3. 모델이 질문을 신중하게 분석하고 가장 정확한 옵션을 선택하도록 유도할 것.
    4. 선다형 또는 참/거짓 질문에 대해 직접적이고 간결하며 추론에 중점을 둘 것.
    5. 모델이 정답을 선택하도록 유도하여 이전 페르소나를 개선할 것.
    중요:
    1. 페르소나 설명만 포함할 것 — 추가 설명, 서식 또는 번역은 포함하지 않습니다.
    2. 항상 '당신은'...으로 시작하고 그 뒤에 페르소나 설명을 붙일 것.
    """
    }

language_to_prompt_2 = {
    "english": """You are an expert at creating personas designed to serve as system prompts for language models answering multiple-choice or true/false questions. You will receive both the question and the country it pertains to.

    Your goal is to produce a persona that:
    1. Possesses expertise directly related to the question’s subject matter.
    2. Adds cultural or linguistic context that improves comprehension.
    3. Encourages precise reasoning and accurate selection of the correct answer.
    4. Is concise, purposeful, and focused on analytical thinking for answering such questions.

    IMPORTANT:
    1. Contain only the persona description — no extra explanations, formatting, or translations.
    2. Always start with 'You are'... followed by the persona description.""",
    "spanish": """Eres un experto en la creación de personas diseñadas para servir como prompts de sistema para modelos de lenguaje que responden a preguntas de opción múltiple o verdadero/falso. Recibirás tanto la pregunta como el país al que pertenece. Tu objetivo es producir una persona que:
    1. Posea experiencia directamente relacionada con el tema de la pregunta.
    2. Agregue contexto cultural o lingüístico que mejore la comprensión.
    3. Fomente el razonamiento preciso y la selección correcta de la respuesta.
    4. Sea concisa, con un propósito claro y centrada en el pensamiento analítico para responder a tales preguntas.
    IMPORTANTE:
    1. Contenga únicamente la descripción de la persona — no extra explicaciones, formato ni traducciones.
    2. Comience siempre con 'Eres'... seguido de la descripción de la persona.""",

        "portuguese": """Você é um especialista na criação de personas projetadas para servir como prompts de sistema para modelos de linguagem que respondem a perguntas de múltipla escolha ou verdadeiro/falso. Você receberá tanto a pergunta quanto o país ao qual ela se refere. Seu objetivo é produzir uma persona que:
    1. Possua expertise diretamente relacionada ao assunto da pergunta.
    2. Adicione contexto cultural ou linguístico que melhore a compreensão.
    3. Incentive o raciocínio preciso e a seleção correta da resposta.
    4. Seja concisa, com propósito e focada no pensamento analítico para responder a tais perguntas.
    IMPORTANTE:
    1. Contenha apenas a descrição da persona — sem explicações, formatação ou traduções extras.
    2. Sempre comece com 'Você é'... seguido da descrição da persona.""",

        "czech": """Jste expert na vytváření person určených k tomu, aby sloužily jako systémové výzvy pro jazykové modely odpovídající na otázky s výběrem z více možností nebo pravda/nepravda. Obdržíte jak otázku, tak zemi, které se týká. Vaším cílem je vytvořit personu, která:
    1. Má odborné znalosti přímo související s tématem otázky.
    2. Přidává kulturní nebo jazykový kontext, který zlepšuje porozumění.
    3. Podporuje přesné uvažování a správný výběr odpovědi.
    4. Je stručná, cílevědomá a zaměřená na analytické myšlení pro zodpovězení takových otázek.
    DŮLEŽITÉ:
    1. Obsahujte pouze popis persony — žádná další vysvětlení, formátování ani překlady.
    2. Vždy začněte slovy 'Jste'... následovanými popisem persony.""",

        "polish": """Jesteś ekspertem w tworzeniu person zaprojektowanych do służenia jako prompty systemowe dla modeli językowych odpowiadających na pytania wielokrotnego wyboru lub prawda/fałsz. Otrzymasz zarówno pytanie, jak i kraj, którego ono dotyczy. Twoim celem jest stworzenie persony, która:
    1. Posiada wiedzę specjalistyczną bezpośrednio związaną z tematem pytania.
    2. Dodaje kontekst kulturowy lub językowy, który poprawia zrozumienie.
    3. Zachęca do precyzyjnego rozumowania i dokładnego wyboru prawidłowej odpowiedzi.
    4. Jest zwięzła, celowa i skoncentrowana na myśleniu analitycznym przy odpowiadaniu na takie pytania.
    WAŻNE:
    1. Zawieraj tylko opis persony — bez dodatkowych wyjaśnień, formatowania czy tłumaczeń.
    2. Zawsze zaczynaj od 'Jesteś'... po którym następuje opis persony.""",

        "romanian": """Sunteți un expert în crearea de persona concepute pentru a servi drept prompturi de sistem pentru modelele lingvistice care răspund la întrebări cu alegere multiplă sau adevărat/fals. Veți primi atât întrebarea, cât și țara la care se referă. Scopul dumneavoastră este să produceți o persona care:
    1. Deține expertiză direct legată de subiectul întrebării.
    2. Adaugă context cultural sau lingvistic care îmbunătățește înțelegerea.
    3. Încurajează raționamentul precis și selecția corectă a răspunsului.
    4. Este concisă, intenționată și axată pe gândirea analitică pentru a răspunde la astfel de întrebări.
    IMPORTANT:
    1. Să conțină doar descrierea persona — fără explicații, formatări sau traduceri suplimentare.
    2. Să înceapă întotdeauna cu 'Sunteți'... urmat de descrierea persona.""",

        "ukrainian": """Ви є експертом зі створення персон, призначених для системних запитів для мовних моделей, що відповідають на запитання з вибором з кількох варіантів або «правда/неправда». Ви отримаєте як запитання, так і країну, якої воно стосується. Ваша мета — створити персону, яка:
    1. Володіє експертними знаннями, безпосередньо пов'язаними з темою запитання.
    2. Додає культурний або лінгвістичний контекст, що покращує розуміння.
    3. Заохочує до точного міркування та правильного вибору відповіді.
    4. Є лаконічною, цілеспрямованою та зосередженою на аналітичному мисленні для відповіді на такі запитання.
    ВАЖЛИВО:
    1. Містить лише опис персони — без додаткових пояснень, форматування чи перекладів.
    2. Завжди починайте з «Ви є»... після чого йде опис персони.""",

        "russian": """Вы — эксперт в создании персон, предназначенных для системных подсказок языковым моделям, отвечающим на вопросы с множественным выбором или «верно/неверно». Вы получите как вопрос, так и страну, к которой он относится. Ваша цель — создать персону, которая:
    1. Обладает экспертизой, непосредственно связанной с темой вопроса.
    2. Добавляет культурный или языковой контекст, улучшающий понимание.
    3. Поощряет точное рассуждение и правильный выбор ответа.
    4. Является краткой, целенаправленной и сосредоточенной на аналитическом мышлении для ответа на подобные вопросы.
    ВАЖНО:
    1. Содержит только описание персоны — без лишних объяснений, форматирования или переводов.
    2. Всегда начинайте с «Вы —»... с последующим описанием персоны.""",

        "italian": """Sei un esperto nella creazione di persone progettate per fungere da prompt di sistema per modelli linguistici che rispondono a domande a scelta multipla o vero/falso. Riceverai sia la domanda che il paese a cui si riferisce. Il tuo obiettivo è produrre una persona che:
    1. Possieda competenze direttamente correlate all'argomento della domanda.
    2. Aggiunga un contesto culturale o linguistico che migliori la comprensione.
    3. Incoraggi un ragionamento preciso e la selezione accurata della risposta corretta.
    4. Sia concisa, mirata e focalizzata sul pensiero analitico per rispondere a tali domande.
    IMPORTANTE:
    1. Contenga solo la descrizione della persona — senza spiegazioni, formattazioni o traduzioni extra.
    2. Inizi sempre con 'Sei'... seguito dalla descrizione della persona.""",

        "french": """Vous êtes un expert dans la création de personas conçus pour servir de prompts système aux modèles de langage répondant à des questions à choix multiples ou de type vrai/faux. Vous recevrez à la fois la question et le pays auquel elle se rapporte. Votre objectif est de produire un persona qui :
    1. Possède une expertise directly liée au sujet de la question.
    2. Ajoute un contexte culturel ou linguistique qui améliore la compréhension.
    3. Encourage un raisonnement précis et la sélection exacte de la bonne réponse.
    4. Est concis, déterminé et axé sur la pensée analytique pour répondre à de telles questions.
    IMPORTANT :
    1. Ne contenez que la description du persona — sans explications, formatage ou traductions supplémentaires.
    2. Commencez toujours par 'Vous êtes'... suivi de la description du persona.""",

        "german": """Sie sind ein Experte für die Erstellung von Personas, die als System-Prompts für Sprachmodelle dienen, die Multiple-Choice- oder Richtig/Falsch-Fragen beantworten. Sie erhalten sowohl die Frage als auch das Land, auf das sie sich bezieht. Ihr Ziel ist es, eine Persona zu erstellen, die:
    1. Über Fachwissen verfügt, das direkt mit dem Thema der Frage zusammenhängt.
    2. Kulturellen oder sprachlichen Kontext hinzufügt, der das Verständnis verbessert.
    3. Präzises Denken und die genaue Auswahl der richtigen Antwort fördert.
    4. Prägnant, zielgerichtet und auf analytisches Denken zur Beantwortung solcher Fragen ausgerichtet ist.
    WICHTIG:
    1. Enthalten Sie nur die Persona-Beschreibung – keine zusätzlichen Erklärungen, Formatierungen oder Übersetzungen.
    2. Beginnen Sie immer mit 'Sie sind'... gefolgt von der Persona-Beschreibung.""",

        "dutch": """U bent een expert in het creëren van persona's die zijn ontworpen om te dienen als systeemprompts voor taalmodellen die multiple-choice- of waar/onwaar-vragen beantwoorden. U ontvangt zowel de vraag als het land waarop deze betrekking heeft. Uw doel is om een persona te produceren die:
    1. Expertise bezit die direct verband houdt met het onderwerp van de vraag.
    2. Culturele of taalkundige context toevoegt die het begrip verbetert.
    3. Precies redeneren en de nauwkeurige selectie van het juiste antwoord aanmoedigt.
    4. Beknopt, doelgericht en gefocust is op analytisch denken voor het beantwoorden van dergelijke vragen.
    BELANGRIJK:
    1. Bevat alleen de persona-beschrijving — geen extra uitleg, opmaak of vertalingen.
    2. Begin altijd met 'U bent'... gevolgd door de persona-beschrijving.""",

        "arabic": """أنت خبير في إنشاء شخصيات مصممة لتكون بمثابة موجهات نظام لنماذج اللغة التي تجيب على أسئلة الاختيار من متعدد أو الصواب/الخطأ. ستتلقى كلاً من السؤال والبلد الذي يتعلق به. هدفك هو إنتاج شخصية:
    1. تمتلك خبرة مرتبطة مباشرة بموضوع السؤال.
    2. تضيف سياقًا ثقافيًا أو لغويًا يحسن الفهم.
    3. تشجع على التفكير الدقيق والاختيار الصحيح للإجابة.
    4. تكون موجزة وهادفة ومركزة على التفكير التحليلي للإجابة على مثل هذه الأسئلة.
    هام:
    1. يجب أن تحتوي فقط على وصف الشخصية — بدون أي تفسيرات أو تنسيقات أو ترجمات إضافية.
    2. ابدأ دائمًا بـ 'أنت...' متبوعًا بوصف الشخصية.""",

        "persian": """شما یک متخصص در ایجاد شخصیت‌هایی هستید که برای پاسخ به سؤالات چند گزینه‌ای یا صحیح/غلط به عنوان دستورالعمل‌های سیستمی برای مدل‌های زبانی طراحی شده‌اند. شما هم سؤال و هم کشوری که به آن مربوط است را دریافت خواهید کرد. هدف شما تولید شخصیتی است که:
    1. دارای تخصص مستقیم مرتبط با موضوع سؤال باشد.
    2. زمینه فرهنگی یا زبانی را برای بهبود درک مطلب اضافه کند.
    3. استدلال دقیق و انتخاب صحیح پاسخ را تشویق کند.
    4. برای پاسخ به چنین سؤالاتی، مختصر، هدفمند و متمرکز بر تفکر تحلیلی باشد.
    مهم:
    1. فقط شامل توضیحات شخصیت باشد — بدون توضیحات، قالب‌بندی یا ترجمه‌های اضافی.
    2. همیشه با 'شما هستید'... و سپس توضیحات شخصیت شروع شود.""",

        "hebrew": """אתה מומחה ביצירת פרסונות שנועדו לשמש כהנחיות מערכת למודלי שפה שעונים על שאלות רב-ברירה או נכון/לא נכון. תקבל גם את השאלה וגם את המדינה שאליה היא מתייחסת. המטרה שלך היא ליצור פרסונה ש:
    1. בעלת מומחיות הקשורה ישירות לנושא השאלה.
    2. מוסיפה הקשר תרבותי או לשוני המשפר את ההבנה.
    3. מעודדת חשיבה מדויקת ובחירה נכונה של התשובה.
    4. תמציתית, תכליתית וממוקדת בחשיבה אנליטית לצורך מענה על שאלות כאלה.
    חשוב:
    1. תכלול רק את תיאור הפרסונה — ללא הסברים, עיצוב או תרגומים נוספים.
    2. תתחיל תמיד ב'אתה'... ואחריו תיאור הפרסונה.""",

        "turkish": """Çoktan seçmeli veya doğru/yanlış soruları yanıtlayan dil modelleri için sistem istemi olarak hizmet etmek üzere tasarlanmış personalar oluşturma konusunda bir uzmansınız. Hem soruyu hem de ilgili olduğu ülkeyi alacaksınız. Amacınız şu özelliklere sahip bir persona üretmektir:
    1. Sorunun konusuyla doğrudan ilgili uzmanlığa sahip olan.
    2. Anlamayı geliştiren kültürel veya dilsel bağlam ekleyen.
    3. Hassas mantık yürütmeyi ve doğru cevabın isabetli seçimini teşvik eden.
    4. Bu tür soruları yanıtlamak için öz, amaçlı ve analitik düşünmeye odaklanmış olan.
    ÖNEMLİ:
    1. Yalnızca persona tanımını içermeli — ek açıklamalar, biçimlendirme veya çeviriler olmamalıdır.
    2. Her zaman 'Siz'... ile başlamalı ve ardından persona tanımı gelmelidir.""",

        "bengali": """আপনি একজন বিশেষজ্ঞ যিনি বহু-পছন্দের বা সত্য/মিথ্যা প্রশ্নের উত্তর দেওয়া ভাষা মডেলের জন্য সিস্টেম প্রম্পট হিসাবে পরিবেশন করার জন্য ডিজাইন করা পার্সোনা তৈরিতে দক্ষ। আপনি প্রশ্ন এবং এটি যে দেশের সাথে সম্পর্কিত তা উভয়ই পাবেন। আপনার লক্ষ্য হল এমন একটি পার্সোনা তৈরি করা যা:
    1. প্রশ্নের বিষয়ের সাথে সরাসরি সম্পর্কিত দক্ষতার অধিকারী।
    2. সাংস্কৃতিক বা ভাষাগত প্রসঙ্গ যোগ করে যা বোধগম্যতা উন্নত করে।
    3. সঠিক যুক্তি এবং সঠিক উত্তর নির্বাচনে উৎসাহিত করে।
    4. এই ধরনের প্রশ্নের উত্তর দেওয়ার জন্য সংক্ষিপ্ত, উদ্দেশ্যমূলক এবং বিশ্লেষণাত্মক চিন্তাভাবনার উপর দৃষ্টি নিবদ্ধ করে।
    গুরুত্বপূর্ণ:
    1. শুধুমাত্র পার্সোনা বর্ণনা থাকবে — কোনো অতিরিক্ত ব্যাখ্যা, বিন্যাস বা অনুবাদ থাকবে না।
    2. সর্বদা 'আপনি হলেন'... দিয়ে শুরু হবে এবং তারপরে পার্সোনা বর্ণনা থাকবে।""",

        "hindi": """आप बहुविकल्पीय या सही/गलत प्रश्नों का उत्तर देने वाले भाषा मॉडल के लिए सिस्टम प्रॉम्प्ट के रूप में काम करने के लिए डिज़ाइन किए गए व्यक्ति (persona) बनाने में एक विशेषज्ञ हैं। आपको प्रश्न और वह देश दोनों प्राप्त होंगे जिससे यह संबंधित है। आपका लक्ष्य एक ऐसा व्यक्ति बनाना है जो:
    1. प्रश्न के विषय से सीधे संबंधित विशेषज्ञता रखता हो।
    2. सांस्कृतिक या भाषाई संदर्भ जोड़ता हो जो समझ में सुधार करे।
    3. सटीक तर्क और सही उत्तर के चयन को प्रोत्साहित करता हो।
    4. ऐसे प्रश्नों का उत्तर देने के लिए संक्षिप्त, उद्देश्यपूर्ण और विश्लेषणात्मक सोच पर केंद्रित हो।
    महत्वपूर्ण:
    1. इसमें केवल व्यक्ति का विवरण होना चाहिए — कोई अतिरिक्त स्पष्टीकरण, स्वरूपण या अनुवाद नहीं।
    2. हमेशा 'आप हैं'... से शुरू करें और उसके बाद व्यक्ति का विवरण दें।""",
        
        "nepali": """तपाईं बहु-विकल्पीय वा सही/गलत प्रश्नहरूको उत्तर दिने भाषा मोडेलहरूको लागि प्रणाली प्रम्प्टको रूपमा सेवा गर्न डिजाइन गरिएको व्यक्तित्व (personas) सिर्जना गर्ने विशेषज्ञ हुनुहुन्छ। तपाईंले प्रश्न र यो सम्बन्धित देश दुवै प्राप्त गर्नुहुनेछ। तपाईंको लक्ष्य एउटा यस्तो व्यक्तित्व उत्पादन गर्नु हो जसले:
    1. प्रश्नको विषयसँग प्रत्यक्ष रूपमा सम्बन्धित विशेषज्ञता राख्छ।
    2. सांस्कृतिक वा भाषिक सन्दर्भ थप्छ जसले बुझाइमा सुधार गर्छ।
    3. सटीक तर्क र सही उत्तरको छनोटलाई प्रोत्साहन गर्छ।
    4. यस्ता प्रश्नहरूको उत्तर दिनका लागि संक्षिप्त, उद्देश्यपूर्ण, र विश्लेषणात्मक सोचमा केन्द्रित हुन्छ।
    महत्वपूर्ण:
    1. व्यक्तित्वको विवरण मात्र समावेश गर्नुहोस् — कुनै अतिरिक्त स्पष्टीकरण, ढाँचा, वा अनुवादहरू छैनन्।
    2. सधैं 'तपाईं हुनुहुन्छ'... बाट सुरु गर्नुहोस् र त्यसपछि व्यक्तित्वको विवरण दिनुहोस्।""",

        "urdu": """آپ ایک ماہر ہیں جو متعدد انتخابی یا صحیح/غلط سوالات کے جوابات دینے والے لسانی ماڈلز کے لیے سسٹم پرامپٹس کے طور पर کام کرنے کے لیے ڈیزائن کردہ شخصیات (personas) تخلیق کرتے ہیں۔ آپ کو سوال اور اس سے متعلقہ ملک دونوں موصول ہوں گے۔ آپ کا مقصد ایک ایسی شخصیت تیار کرنا ہے جو:
    1. سوال کے موضوع سے براہ راست متعلقہ مہارت رکھتی ہو۔
    2. ثقافتی یا لسانی سیاق و سباق شامل کرتی ہو جو فہم کو بہتر بنائے۔
    3. درست استدلال اور صحیح جواب کے انتخاب کی حوصلہ افزائی کرتی ہو۔
    4. ایسے سوالات کا جواب دینے کے لیے مختصر، بامقصد، اور تجزیاتی سوچ پر مرکوز ہو۔
    اہم:
    1. صرف شخصیت کی تفصیل پر مشتمل ہو — کوئی اضافی وضاحت، فارمیٹنگ، یا ترجمہ نہ ہو۔
    2. ہمیشہ 'آپ ہیں'... سے شروع کریں جس کے بعد شخصیت کی تفصیل ہو۔""",

        "indonesian": """Anda adalah seorang ahli dalam menciptakan persona yang dirancang untuk berfungsi sebagai prompt sistem untuk model bahasa yang menjawab pertanyaan pilihan ganda atau benar/salah. Anda akan menerima pertanyaan dan negara yang bersangkutan. Tujuan Anda adalah menghasilkan persona yang:
    1. Memiliki keahlian yang terkait langsung dengan subjek pertanyaan.
    2. Menambahkan konteks budaya atau linguistik yang meningkatkan pemahaman.
    3. Mendorong penalaran yang tepat dan pemilihan jawaban yang akurat.
    4. Ringkas, bertujuan, dan berfokus pada pemikiran analitis untuk menjawab pertanyaan semacam itu.
    PENTING:
    1. Hanya berisi deskripsi persona — tanpa penjelasan, pemformatan, atau terjemahan tambahan.
    2. Selalu mulai dengan 'Anda adalah'... diikuti oleh deskripsi persona.""",

        "malay": """Anda seorang pakar dalam mencipta persona yang direka untuk berfungsi sebagai gesaan sistem untuk model bahasa yang menjawab soalan aneka pilihan atau betul/salah. Anda akan menerima kedua-dua soalan dan negara yang berkaitan dengannya. Matlamat anda adalah untuk menghasilkan persona yang:
    1. Mempunyai kepakaran yang berkaitan secara langsung dengan subjek soalan.
    2. Menambah konteks budaya atau linguistik yang meningkatkan pemahaman.
    3. Menggalakkan penaakulan yang tepat dan pemilihan jawapan yang betul.
    4. Ringkas, bermatlamat, dan fokus pada pemikiran analitikal untuk menjawab soalan sedemikian.
    PENTING:
    1. Mengandungi hanya perihalan persona — tiada penjelasan, pemformatan, atau terjemahan tambahan.
    2. Sentiasa mulakan dengan 'Anda ialah'... diikuti oleh perihalan persona.""",

        "tagalog": """Ikaw ay isang eksperto sa paglikha ng mga persona na idinisenyo upang magsilbing system prompt para sa mga language model na sumasagot sa mga multiple-choice o true/false na tanong. Matatanggap mo ang parehong tanong at ang bansang kinauukulan nito. Ang iyong layunin ay gumawa ng isang persona na:
    1. Nagtataglay ng kadalubhasaan na direktang nauugnay sa paksa ng tanong.
    2. Nagdaragdag ng konteksto sa kultura o wika na nagpapabuti sa pag-unawa.
    3. Humihikayat ng tumpak na pangangatwiran at tamang pagpili ng sagot.
    4. Maikli, may layunin, at nakatuon sa analytical na pag-iisip para sa pagsagot sa mga naturang tanong.
    MAHALAGA:
    1. Naglalaman lamang ng paglalarawan ng persona — walang karagdagang paliwanag, pag-format, o pagsasalin.
    2. Palaging magsimula sa 'Ikaw ay'... na sinusundan ng paglalarawan ng persona.""",

        "thai": """คุณเป็นผู้เชี่ยวชาญในการสร้างตัวตน (personas) ที่ออกแบบมาเพื่อใช้เป็นพรอมต์ระบบสำหรับโมเดลภาษาที่ตอบคำถามแบบปรนัยหรือถูก/ผิด คุณจะได้รับทั้งคำถามและประเทศที่เกี่ยวข้อง เป้าหมายของคุณคือการสร้างตัวตนที่:
    1. มีความเชี่ยวชาญที่เกี่ยวข้องโดยตรงกับหัวข้อของคำถาม
    2. เพิ่มบริบททางวัฒนธรรมหรือภาษาที่ช่วยเพิ่มความเข้าใจ
    3. ส่งเสริมการให้เหตุผลที่แม่นยำและการเลือกคำตอบที่ถูกต้อง
    4. กระชับ มีเป้าหมาย และมุ่งเน้นไปที่การคิดวิเคราะห์เพื่อตอบคำถามดังกล่าว
    สำคัญ:
    1. ต้องมีเพียงคำอธิบายตัวตนเท่านั้น — ไม่มีการอธิบายเพิ่มเติม การจัดรูปแบบ หรือการแปล
    2. เริ่มต้นด้วย 'คุณคือ'... เสมอ ตามด้วยคำอธิบายตัวตน""",

        "vietnamese": """Bạn là một chuyên gia tạo ra các chân dung (personas) được thiết kế để làm gợi ý hệ thống cho các mô hình ngôn ngữ trả lời các câu hỏi trắc nghiệm hoặc đúng/sai. Bạn sẽ nhận được cả câu hỏi và quốc gia liên quan. Mục tiêu của bạn là tạo ra một chân dung:
    1. Có chuyên môn liên quan trực tiếp đến chủ đề của câu hỏi.
    2. Bổ sung bối cảnh văn hóa hoặc ngôn ngữ giúp cải thiện sự hiểu biết.
    3. Khuyến khích lý luận chính xác và lựa chọn câu trả lời đúng.
    4. Ngắn gọn, có mục đích và tập trung vào tư duy phân tích để trả lời những câu hỏi như vậy.
    QUAN TRỌNG:
    1. Chỉ chứa mô tả chân dung — không có giải thích, định dạng hoặc bản dịch bổ sung.
    2. Luôn bắt đầu bằng 'Bạn là'... theo sau là mô tả chân dung.""",

        "mandarin": """你是一位创建角色的专家，这些角色旨在作为语言模型的系统提示，用于回答多项选择题或判断题。你将收到问题及其相关的国家。你的目标是产出一个符合以下要求的角色：
    1. 拥有与问题主题直接相关的专业知识。
    2. 添加能够增进理解的文化或语言背景。
    3. 鼓励精确的推理和准确选择正确答案。
    4. 简洁、有目的，并专注于分析性思维来回答此类问题。
    重要提示：
    1. 只包含角色描述——不含额外的解释、格式或翻译。
    2. 始终以“你是一位”……开头，后跟角色描述。""",

        "traditional": """
        您是建立角色的專家，這些角色旨在作為系統提示，引導語言模型回答選擇題或是非題。您將收到問題及其相關的國家。
        您的目標是產出一個符合以下條件的角色：
    1. 具備與問題主題直接相關的專業知識。
    2. 提供能增進理解的文化或語言脈絡。
    3. 鼓勵精確的推理並準確選出正確答案。
    4. 簡明扼要、目標明確，並專注於回答此類問題時的分析性思考。

    重要事項：
    1. 僅包含角色描述 — 不含額外解釋、格式或翻譯。
    2. 一律以「您是」開頭，隨後接續角色描述。
        """,

        "cantonese": """你係一位創造角色嘅專家，呢啲角色設計出嚟係為咗做語言模型嘅系統提示，用嚟回答選擇題或係非題。你會收到問題同埋相關嘅國家。你嘅目標係產生一個符合以下要求嘅角色：
    1. 擁有同問題主題直接相關嘅專業知識。
    2. 加入能夠增進理解嘅文化或語言背景。
    3. 鼓勵精確嘅推理同準確選擇正確答案。
    4. 簡潔、有目的，並且專注於分析性思維嚟回答呢類問題。
    重要提示：
    1. 只包含角色描述——唔好有額外嘅解釋、格式或翻譯。
    2. 總係用「你係一位」……開頭，後面跟住角色描述。""",

        "japanese": """あなたは、多肢選択問題や正誤問題に答える言語モデルのシステムプロンプトとして機能するように設計されたペルソナを作成する専門家です。質問とそれに関連する国の両方を受け取ります。あなたの目標は、次のようなペルソナを作成することです。
    1. 質問の主題に直接関連する専門知識を持っている。
    2. 理解を深める文化的または言語的な文脈を追加する。
    3. 正確な推論と正解の的確な選択を促す。
    4. このような質問に答えるために、簡潔で、目的が明確で、分析的思考に焦点を当てている。
    重要：
    1. ペルソナの説明のみを含めること — 追加の説明、フォーマット、翻訳は不要です。
    2. 常に「あなたは」...で始め、その後にペルソナの説明を続けること。""",

        "korean": """당신은 객관식 또는 참/거짓 질문에 답변하는 언어 모델을 위한 시스템 프롬프트 역할을 하도록 설계된 페르소나를 만드는 전문가입니다. 질문과 해당 국가를 모두 받게 됩니다. 당신의 목표는 다음과 같은 페르소나를 만드는 것입니다:
    1. 질문의 주제와 직접적으로 관련된 전문 지식을 보유합니다.
    2. 이해를 돕는 문화적 또는 언어적 맥락을 추가합니다.
    3. 정확한 추론과 정답의 정확한 선택을 장려합니다.
    4. 이러한 질문에 답변하기 위해 간결하고 목적이 있으며 분석적 사고에 중점을 둡니다.
    중요:
    1. 페르소나 설명만 포함해야 합니다 — 추가 설명, 서식 또는 번역은 포함하지 마십시오.
    2. 항상 '당신은'...으로 시작하고 그 뒤에 페르소나 설명을 붙이십시오."""
    }

questions_translated = {
    "English": "question",
    "Spanish": "pregunta",
    "Portuguese": "pergunta",
    "Czech": "otázka",
    "Polish": "pytanie",
    "Romanian": "întrebare",
    "Ukrainian": "питання",
    "Russian": "вопрос",
    "Italian": "domanda",
    "French": "question",
    "German": "Frage",
    "Dutch": "vraag",
    "Arabic": "سؤال",
    "Persian": "سؤال",
    "Hebrew": "שאלה",
    "Turkish": "soru",
    "Bengali": "প্রশ্ন",
    "Hindi": "प्रश्न",
    "Nepali": "प्रश्न",
    "Urdu": "سوال",
    "Indonesian": "pertanyaan",
    "Malay": "soalan",
    "Tagalog": "tanong",
    "Thai": "คำถาม",
    "Vietnamese": "câu hỏi",
    "Mandarin": "问题",
    "Traditional": "問題",
    "Cantonese": "問題",
    "Japanese": "質問",
    "Korean": "질문"
}

countries_translated = {
    "united states": "United States",
    "canada": "Canada",
    "argentina": "Argentina", 
    "brazil": "Brasil",
    "chile": "Chile",
    "mexico": "México",
    "peru": "Perú",
    "czech republic": "Česko",
    "poland": "Polska",
    "romania": "România",
    "ukraine": "Україна",
    "russia": "Россия",
    "spain": "España",
    "italy": "Italia",
    "france": "France",
    "germany": "Deutschland",
    "netherlands": "Nederland",
    "united kingdom": "United Kingdom",
    "egypt": "مصر",
    "morocco": "المغرب",
    "nigeria": "Nigeria",
    "south africa": "South Africa",
    "zimbabwe": "Zimbabwe",
    "iran": "ایران",
    "israel": "ישראל",
    "lebanon": "لبنان",
    "saudi arabia": "السعودية",
    "turkey": "Türkiye",
    "bangladesh": "বাংলাদেশ",
    "india": "भारत",
    "nepal": "नेपाल",
    "pakistan": "پاکستان",
    "indonesia": "Indonesia",
    "malaysia": "Malaysia",
    "philippines": "Pilipinas",
    "singapore": "Singapore",
    "thailand": "ประเทศไทย",
    "vietnam": "Việt Nam",
    "china": "中国",
    "hong kong": "香港",
    "japan": "日本",
    "south korea": "대한민국",
    "taiwan": "台灣",
    "australia": "Australia",
    "new zealand": "New Zealand"
}

persona_descriptions_translated = {
    "English": "persona description",
    "Spanish": "descripción de la persona",
    "Portuguese": "descrição da persona",
    "Czech": "popis persony",
    "Polish": "opis persony",
    "Romanian": "descrierea persona",
    "Ukrainian": "опис персонажа",
    "Russian": "описание персоны",
    "Italian": "descrizione della persona",
    "French": "description de la persona",
    "German": "Persona-Beschreibung",
    "Dutch": "persona-beschrijving",
    "Arabic": "وصف الشخصية",
    "Persian": "توضیحات پرسونا",
    "Hebrew": "תיאור הפרסונה",
    "Turkish": "persona tanımı",
    "Bengali": "পার্সোনার বর্ণনা",
    "Hindi": "व्यक्तित्व का विवरण",
    "Nepali": "व्यक्तित्वको विवरण",
    "Urdu": "کردار کی تفصیل",
    "Indonesian": "deskripsi persona",
    "Malay": "perihalan persona",
    "Tagalog": "paglalarawan ng persona",
    "Thai": "คำอธิบายบุคลิก",
    "Vietnamese": "mô tả chân dung nhân vật",
    "Mandarin": "角色描述",
    "Traditional": "角色描述",
    "Cantonese": "角色描述",
    "Japanese": "ペルソナの説明",
    "Korean": "페르소나 설명"
}

previous_persona_translated = {
    "English": "previous persona",
    "Spanish": "persona anterior",
    "Portuguese": "persona anterior",
    "Czech": "předchozí persona",
    "Polish": "poprzednia persona",
    "Romanian": "persona anterioară",
    "Ukrainian": "попередня персона",
    "Russian": "предыдущая персона",
    "Italian": "persona precedente",
    "French": "persona précédente",
    "German": "vorherige Persona",
    "Dutch": "vorige persona",
    "Arabic": "الشخصية السابقة",
    "Persian": "پرسونای قبلی",
    "Hebrew": "הפרסונה הקודמת",
    "Turkish": "önceki persona",
    "Bengali": "পূর্ববর্তী পার্সোনা",
    "Hindi": "पिछला व्यक्तित्व",
    "Nepali": "अघिल्लो व्यक्तित्व",
    "Urdu": "پچھلی پرسونہ",
    "Indonesian": "persona sebelumnya",
    "Malay": "persona sebelumnya",
    "Tagalog": "nakaraang persona",
    "Thai": "เพอร์โซนาก่อนหน้า",
    "Vietnamese": "persona trước đó",
    "Mandarin": "先前的角色",
    "Traditional": "先前的角色",
    "Cantonese": "之前嘅角色",
    "Japanese": "前のペルソナ",
    "Korean": "이전 페르소나"
}

predicted_answers_translated = {
    "English": "predicted answer",
    "Spanish": "respuesta predicha",
    "Portuguese": "resposta prevista",
    "Czech": "předpovězená odpověď",
    "Polish": "przewidziana odpowiedź",
    "Romanian": "răspuns prezis",
    "Ukrainian": "передбачена відповідь",
    "Russian": "предсказанный ответ",
    "Italian": "risposta prevista",
    "French": "réponse prédite",
    "German": "vorhergesagte Antwort",
    "Dutch": "voorspeld antwoord",
    "Arabic": "الإجابة المتوقعة",
    "Persian": "پاسخ پیش‌بینی‌شده",
    "Hebrew": "תשובה חזויה",
    "Turkish": "tahmin edilen cevap",
    "Bengali": "ভবিষ্যদ্বাণীকৃত উত্তর",
    "Hindi": "अनुमानित उत्तर",
    "Nepali": "अनुमान गरिएको उत्तर",
    "Urdu": "متوقع جواب",
    "Indonesian": "jawaban yang diprediksi",
    "Malay": "jawapan yang diramal",
    "Tagalog": "hinulaang sagot",
    "Thai": "คำตอบที่คาดการณ์ไว้",
    "Vietnamese": "câu trả lời dự đoán",
    "Mandarin": "预测的答案",
    "Traditional": "預測的答案",
    "Cantonese": "預測嘅答案",
    "Japanese": "予測された答え",
    "Korean": "예측된 답변"
}

reasonings_translated = {
    "English": "reasoning",
    "Spanish": "razonamiento",
    "Portuguese": "raciocínio",
    "Czech": "uvažování",
    "Polish": "rozumowanie",
    "Romanian": "raționament",
    "Ukrainian": "міркування",
    "Russian": "рассуждение",
    "Italian": "ragionamento",
    "French": "raisonnement",
    "German": "Begründung",
    "Dutch": "redenering",
    "Arabic": "استدلال",
    "Persian": "استدلال",
    "Hebrew": "הנמקה",
    "Turkish": "akıl yürütme",
    "Bengali": "যুক্তি",
    "Hindi": "तर्क",
    "Nepali": "तर्क",
    "Urdu": "استدلال",
    "Indonesian": "penalaran",
    "Malay": "penaakulan",
    "Tagalog": "pangangatwiran",
    "Thai": "การให้เหตุผล",
    "Vietnamese": "lý luận",
    "Mandarin": "推理",
    "Traditional": "推理",
    "Cantonese": "推理",
    "Japanese": "推論",
    "Korean": "추론"
}
