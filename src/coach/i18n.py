"""Fixed texts per language, used without an LLM call."""

STRINGS = {
    "feedback_intro": {
        "de": "Danke, das Übungsgespräch ist jetzt zu Ende. Hier ist dein Feedback:",
        "fr": "Merci, l'entretien d'entraînement est maintenant terminé. Voici ton feedback :",
        "it": "Grazie, il colloquio di prova è finito. Ecco il tuo feedback:",
    },
    "finished": {
        "de": "Das Gespräch ist beendet. Starte einen neuen Chat, wenn du nochmals üben möchtest.",
        "fr": "L'entretien est terminé. Ouvre une nouvelle conversation pour t'entraîner encore.",
        "it": "Il colloquio è finito. Apri una nuova chat se vuoi esercitarti ancora.",
    },
    "ack": {
        "de": [
            "Danke.",
            "Okay, danke.",
            "Verstehe.",
            "Danke für deine Antwort.",
            "Alles klar, danke.",
            "Gut, dann weiter.",
        ],
        "fr": [
            "Merci.",
            "D'accord, merci.",
            "Je vois.",
            "Merci pour ta réponse.",
            "Très bien, merci.",
            "Bien, continuons.",
        ],
        "it": [
            "Grazie.",
            "Va bene, grazie.",
            "Capisco.",
            "Grazie per la risposta.",
            "D'accordo, grazie.",
            "Bene, andiamo avanti.",
        ],
    },
    "referral": {
        "de": "Wenn dich die Bewerbungen gerade belasten, sprich mit einer Person, der du vertraust: deiner Klassenlehrperson, deinen Eltern oder der Berufsberatung im BIZ. Du kannst auch jederzeit gratis und vertraulich die 147 von Pro Juventute anrufen oder ihnen schreiben. Du musst das nicht allein schaffen.",
        "fr": "Si les candidatures te pèsent en ce moment, parles-en à une personne de confiance : ton enseignant·e, tes parents ou l'orientation professionnelle. Tu peux aussi appeler ou écrire à tout moment, gratuitement et en toute confidentialité, au 147 de Pro Juventute. Tu n'as pas à gérer ça seul·e.",
        "it": "Se le candidature ti pesano in questo periodo, parlane con una persona di fiducia: il tuo docente, i tuoi genitori o l'orientamento professionale. Puoi anche chiamare o scrivere in qualsiasi momento, gratis e in modo confidenziale, al 147 di Pro Juventute. Non devi affrontarlo da solo o da sola.",
    },
    "encourage": {
        "de": "Danke, dass du das so offen sagst. Absagen sind belastend, und sehr viele Jugendliche erleben das. Hier üben wir in Ruhe, ohne Druck. Wenn dich das beschäftigt, kannst du auch mit deiner Klassenlehrperson oder der Berufsberatung im BIZ darüber reden.",
        "fr": "Merci de le dire aussi ouvertement. Les refus sont difficiles, et beaucoup de jeunes vivent la même chose. Ici, on s'entraîne tranquillement, sans pression. Si ça te préoccupe, tu peux aussi en parler à ton enseignant·e ou à l'orientation professionnelle.",
        "it": "Grazie di dirlo così apertamente. I rifiuti pesano, e moltissimi giovani li vivono. Qui ci esercitiamo con calma, senza pressione. Se ti preoccupa, puoi parlarne anche con il tuo docente o con l'orientamento professionale.",
    },
    "helpline": {
        "de": "Du kannst auch jederzeit gratis und vertraulich die 147 von Pro Juventute anrufen oder ihnen schreiben.",
        "fr": "Tu peux aussi appeler ou écrire à tout moment, gratuitement et en toute confidentialité, au 147 de Pro Juventute.",
        "it": "Puoi anche chiamare o scrivere in qualsiasi momento, gratis e in modo confidenziale, al 147 di Pro Juventute.",
    },
    "phone_pickup": {
        "de": "{company}, {name}, grüezi. Was kann ich für dich tun?",
        "fr": "{company}, {name}, bonjour. Que puis-je faire pour toi ?",
        "it": "{company}, {name}, buongiorno. Cosa posso fare per te?",
    },
    "intake": {
        "de": "Hallo! Ich bin dein Coach für Vorstellungsgespräche. Für welche Lehrstelle möchtest du heute üben? (Mit «Stopp» kannst du jederzeit aufhören und bekommst dein Feedback.)",
        "fr": "Bonjour ! Je suis ton coach pour les entretiens d'embauche. Pour quel apprentissage aimerais-tu t'entraîner aujourd'hui ? (Avec « stop », tu peux arrêter à tout moment et recevoir ton feedback.)",
        "it": "Ciao! Sono il tuo coach per i colloqui di lavoro. Per quale apprendistato vuoi esercitarti oggi? (Con «basta» puoi smettere in qualsiasi momento e ricevere il tuo feedback.)",
    },
    "general_ok": {
        "de": "Kein Problem, das geht vielen so. Dann üben wir ein allgemeines Vorstellungsgespräch.",
        "fr": "Pas de problème, c'est le cas de beaucoup de jeunes. On s'entraîne alors avec un entretien général.",
        "it": "Nessun problema, succede a tanti. Allora ci esercitiamo con un colloquio generale.",
    },
    "undecided_tip": {
        "de": "Ein Tipp für die nächsten Wochen: Überlege dir, welche Lehrstelle dich am meisten interessiert. Die Berufsberatung im BIZ hilft dir dabei, und mit einem konkreten Beruf kannst du dich noch gezielter vorbereiten.",
        "fr": "Un conseil pour les prochaines semaines : réfléchis à l'apprentissage qui t'intéresse le plus. L'orientation professionnelle peut t'aider, et avec un métier précis, tu pourras te préparer de façon encore plus ciblée.",
        "it": "Un consiglio per le prossime settimane: pensa a quale apprendistato ti interessa di più. L'orientamento professionale ti può aiutare, e con una professione precisa potrai prepararti in modo ancora più mirato.",
    },
    "general_occupation": {
        "de": "eine Lehrstelle (Beruf noch offen)",
        "fr": "un apprentissage (métier pas encore choisi)",
        "it": "un apprendistato (professione non ancora scelta)",
    },
    "short_interview": {
        "de": "Wir haben nur kurz geübt ({n} Antworten). Die Bewertung ist deshalb nur ein erster Eindruck; für die meisten Kriterien gab es noch nichts zu bewerten.",
        "fr": "Nous ne nous sommes entraînés que brièvement ({n} réponses). L'évaluation n'est donc qu'une première impression ; pour la plupart des critères, il n'y avait encore rien à évaluer.",
        "it": "Ci siamo esercitati solo brevemente ({n} risposte). La valutazione è quindi solo una prima impressione; per la maggior parte dei criteri non c'era ancora niente da valutare.",
    },
    "repeat_question": {
        "de": "Das hast du mir vorhin schon erzählt. Meine Frage war aber eine andere: {q}",
        "fr": "Tu me l'as déjà raconté tout à l'heure. Ma question était différente : {q}",
        "it": "Me l'hai già raccontato prima. La mia domanda però era un'altra: {q}",
    },
    "lang_name": {"de": "German (Standard German, Swiss spelling: ss instead of ß)", "fr": "French", "it": "Italian"},
    "address": {
        "informal": {"de": "du", "fr": "tu", "it": "tu"},
        "formal": {"de": "Sie", "fr": "vous", "it": "Lei"},
    },
}


def t(key: str, lang: str) -> str:
    return STRINGS[key][lang]
