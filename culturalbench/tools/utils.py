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

language_to_prompt = {
    "english": """You are an expert in crafting personas that will be used as a system prompt to a language model in answering a multiple-choice or true/false question accurately. You will be given the question and the relevant country that the question content is related to.

    Your task is to generate a new persona that MUST:
    1. Have expertise directly RELEVANT to the question content.
    2. Provide cultural or linguistic context that enhances understanding.
    3. Guide the model to analyze the question carefully and choose the most accurate option.
    4. Be direct, concise, and focused on reasoning for multiple-choice or true-false questions.

    IMPORTANT:
    1. Contain only the persona description — no extra explanations, formatting, or translations.
    2. Always start with 'You are'... followed by the persona description.""",
        "spanish": """
    Eres un experto en crear personas que se utilizarán como prompt del sistema para un modelo de lenguaje al responder preguntas de opción múltiple o verdadero/falso con precisión. Se le dará la pregunta y el país relevante al que se relaciona el contenido de la pregunta. Su tarea es generar una nueva persona que DEBE:
    1. Tener experiencia directamente RELEVANTE al contenido de la pregunta.
    2. Proporcionar contexto cultural o lingüístico que mejore la comprensión.
    3. Guiar al modelo para analizar la pregunta cuidadosamente y elegir la opción más precisa.
    4. Ser directo, conciso y centrado en el razonamiento para preguntas de opción múltiple o verdadero/falso.
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
    중요:
    1. 페르소나 설명만 포함할 것 — 추가 설명, 서식 또는 번역은 포함하지 않습니다.
    2. 항상 '당신은'...으로 시작하고 그 뒤에 페르소나 설명을 붙일 것.
    """
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

persona_translated = {
    "English": "persona",
    "Spanish": "persona",
    "Portuguese": "persona",
    "Czech": "persona",
    "Polish": "persona",
    "Romanian": "persona",
    "Ukrainian": "персона",
    "Russian": "персона",
    "Italian": "persona",
    "French": "persona",
    "German": "Persona",
    "Dutch": "persona",
    "Arabic": "الشخصية",
    "Persian": "پرسونا",
    "Hebrew": "פרסונה",
    "Turkish": "persona",
    "Bengali": "পার্সোনা",
    "Hindi": "व्यक्तित्व",
    "Nepali": "व्यक्तित्व",
    "Urdu": "پرسونہ",
    "Indonesian": "persona",
    "Malay": "persona",
    "Tagalog": "persona",
    "Thai": "เพอร์โซนา",
    "Vietnamese": "persona",
    "Mandarin": "角色",
    "Traditional": "角色",
    "Cantonese": "角色",
    "Japanese": "ペルソナ",
    "Korean": "페르소나"
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
