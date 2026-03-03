
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal


class UserRequest(BaseModel):
    user_prompt: str = Field(..., description="User's story prompt")
    story_length: Optional[str] = Field("medium", description="short/medium/long")
    max_branches: Optional[int] = Field(3, description="Maximum story branches (routes)")
    is_part_of_other_universe: Optional[bool] = Field(False, description="Part of existing universe")
    tone: Optional[str] = Field("balanced", description="Story tone")
    general_artstyle: Optional[str] = Field("anime", description="Art style preference")


class Setting(BaseModel):
    setting: str = Field(..., description="Detailed world setting and rules")
    genre: Optional[str] = Field(None, description="Primary genre")
    time_period: Optional[str] = Field(None, description="Time period or era")
    world_rules: Optional[str] = Field(None, description="Special world rules or mechanics")


class CharacterNode(BaseModel):
    id: str = Field(..., description="Character identifier (name)")
    label: str = Field(..., description="Character display name")


class CharacterEdge(BaseModel):
    source: str = Field(..., description="Source character id")
    target: str = Field(..., description="Target character id")
    label: Literal["friends", "enemies", "love interest"] = Field(..., description="Relationship type")
    directed: bool = Field(True, description="Is directed edge")


class CharacterGraph(BaseModel):
    nodes: List[CharacterNode] = Field(..., description="List of characters")
    edges: List[CharacterEdge] = Field(..., description="List of relationships")


class CharacterAppearance(BaseModel):
    descriptions: List[str] = Field(..., description="Visual descriptions for each character")


class CharacterDetails(BaseModel):
    id: int = Field(..., description="Character ID")
    name: str = Field(..., description="Character name")
    role: Literal["protagonist", "antagonist", "supporting", "minor"] = Field(..., description="Character role")
    importance: Literal["major", "minor"] = Field(..., description="Story importance")
    archetype: str = Field(..., description="Character archetype")
    motive: str = Field(..., description="Character motivation")
    relationship: str = Field(..., description="Key relationships")
    background: str = Field(..., description="Character backstory")
    dialogue_style: str = Field(..., description="How character speaks")
    secrets: List[str] = Field(default_factory=list, description="Character secrets")


class LocationNode(BaseModel):
    id: str = Field(..., description="Location identifier")
    label: str = Field(..., description="Location display name")
    type: Literal[
        "indoor",
        "outdoor",
        "dungeon",
        "city",
        "wilderness",
        "special",
    ] = Field(..., description="Location type")

    @validator("type", pre=True)
    def validate_type(cls, v):
        allowed = {"indoor", "outdoor", "dungeon", "city", "wilderness", "special"}
        if v not in allowed:
            raise ValueError(f"Invalid location type: {v}")
        return v


class LocationEdge(BaseModel):
    source: str = Field(..., description="Source location id")
    target: str = Field(..., description="Target location id")
    label: Literal["direct_path", "transport", "teleport", "conditional", "secret"] = Field(
        ..., description="Connection type"
    )
    directed: bool = Field(True, description="Is directed edge")
    bidirectional: bool = Field(False, description="Can travel both ways")


class LocationGraph(BaseModel):
    nodes: List[LocationNode] = Field(..., description="List of locations")
    edges: List[LocationEdge] = Field(..., description="List of connections")


class LocationDescription(BaseModel):
    descriptions: List[str] = Field(..., description="Visual descriptions for each location")


class LocationAffordance(BaseModel):
    location: str = Field(..., description="Location name")
    kind: Literal["indoor", "outdoor", "mixed"] = Field(..., description="Physical kind")
    enterable: bool = Field(..., description="Whether characters can be inside a physical interior here")
    scale: Literal["object", "room", "building", "area"] = Field(..., description="Approximate scale")
    notes: Optional[str] = Field(None, description="Short notes / prohibitions")


class LocationDetails(BaseModel):
    id: int = Field(..., description="Location ID")
    name: str = Field(..., description="Location name")
    character_list: List[str] = Field(default_factory=list, description="Characters present")
    description: str = Field(..., description="Detailed description")
    story_importance: str = Field(..., description="Importance to story")
    items: Optional[List[str]] = Field(default_factory=list, description="Important items in location")


class StoryOutline(BaseModel):
    ellipsis: str = Field(..., description="Story synopsis/outline")


class StoryNode(BaseModel):
    id: str = Field(..., description="Node identifier")
    label: str = Field(..., description="Node label")
    type: Literal["character_group", "key_interaction", "character_state", "plot_point"] = Field(
        ..., description="Node type"
    )
    characters: List[str] = Field(..., description="Characters involved")
    location: str = Field(..., description="Location of event")
    description: str = Field(..., description="Event description")


class StoryEdge(BaseModel):
    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")
    label: Literal[
        "action", "decision", "revelation", "conflict", "transformation", "external_event"
    ] = Field(..., description="Edge type")
    description: str = Field(..., description="Transition description")
    intensity: Literal["low", "medium", "high"] = Field(..., description="Narrative impact")


class StoryGraph(BaseModel):
    nodes: List[StoryNode] = Field(..., description="Story nodes")
    edges: List[StoryEdge] = Field(..., description="Story connections")



class OutlineBeat(BaseModel):
    id: str = Field(..., description="Beat identifier, e.g. 'beat_01'")
    act: int = Field(..., description="Act number (e.g. 1, 2, 3)")
    order: int = Field(..., description="Global order index of the beat")
    title: str = Field(..., description="Short descriptive beat title")
    summary: str = Field(..., description="1–3 sentence summary of the beat")
    tension_level: Literal["low", "medium", "high"] = Field(..., description="Narrative tension level")
    purpose: str = Field(..., description="Narrative purpose, e.g. 'introduction', 'climax', etc.")


class StoryOutlineFull(BaseModel):
    theory: str = Field(..., description="Narrative theory used (e.g., 'three_act', 'freytag')")
    beats: List[OutlineBeat] = Field(..., description="Ordered beats of the story")


class SceneContract(BaseModel):
    id: str = Field(..., description="Scene identifier, e.g. 'scene_001'")
    beat_id: str = Field(..., description="ID of the outline beat this scene realizes")
    location: str = Field(..., description="Location name for this scene")
    pov_character: str = Field(..., description="POV character for this scene")
    present_characters: List[str] = Field(..., description="Characters present in the scene")
    summary: str = Field(..., description="1–3 sentence summary of what happens in this scene")

    branch_id: str = Field("main", description="Route this scene belongs to: 'main' or 'branch_xx'")
    branch_order: int = Field(0, description="Order of this scene within its branch")


class SceneLine(BaseModel):
    type: Literal["dialogue", "narration", "thought"] = Field(..., description="Line type")
    speaker: Optional[str] = Field(None, description="Speaker name for dialogue/thought, None for narration")
    text: str = Field(..., description="Actual text of the line")


class SceneChoiceOption(BaseModel):
    id: str = Field(..., description="Option identifier within the choice, e.g. 'opt_01'")
    text: str = Field(..., description="Button text shown to the player")
    leads_to_scene_id: str = Field(..., description="Target scene ID if this option is chosen")
    leads_to_branch_id: Optional[str] = Field(
        None,
        description="If set, switch to this branch for subsequent scenes (e.g. 'branch_01')",
    )
    is_fake: bool = Field(
        False,
        description="True if this option does NOT change route/graph (cosmetic choice)",
    )


class SceneChoice(BaseModel):
    id: str = Field(..., description="Choice identifier within the scene, e.g. 'choice_01'")
    appears_after_line: int = Field(
        ...,
        description="Index in 'lines' AFTER which this menu appears (0 = after first line)",
    )
    options: List[SceneChoiceOption] = Field(..., description="List of choice options")


class SceneScript(BaseModel):
    scene_id: str = Field(..., description="ID of the scene this script belongs to")
    lines: List[SceneLine] = Field(..., description="Ordered list of lines in the scene")
    summary: str = Field(..., description="Short summary of the scene (for RAG / recall)")

    branch_id: str = Field("main", description="Route this scene belongs to")
    branch_order: int = Field(0, description="Order of this scene within its branch")
    choices: List[SceneChoice] = Field(default_factory=list, description="Interactive menus in this scene")


class MasterContext(BaseModel):

    user_request: Optional[UserRequest] = None

    char_list: List[str] = Field(default_factory=list, description="Character names")
    char_graph: Optional[CharacterGraph] = None
    char_appearance: Optional[CharacterAppearance] = None

    loc_list: List[str] = Field(default_factory=list, description="Location names")
    loc_graph: Optional[LocationGraph] = None
    loc_description: Optional[LocationDescription] = None

    setting: Optional[Setting] = None
    ellipsis: Optional[StoryOutline] = None
    story_graph: Optional[StoryGraph] = None

    class Config:
        arbitrary_types_allowed = True


#image meta

class CharacterImage(BaseModel):
    id: str = Field(..., description="Unique image id (per generation), e.g. 'char_001'")
    character: str = Field(..., description="Character name this sprite belongs to")
    path: str = Field(..., description="Filesystem path or URL to the image")
    aspect_ratio: str = Field("1:1", description="Aspect ratio, e.g. '1:1'")


class LocationImage(BaseModel):
    id: str = Field(..., description="Unique image id (per generation), e.g. 'bg_001'")
    location: str = Field(..., description="Location name this background belongs to")
    path: str = Field(..., description="Filesystem path or URL to the image")
    aspect_ratio: str = Field("16:9", description="Aspect ratio, e.g. '16:9'")



class BranchSpec(BaseModel):
    id: str = Field(..., description="Branch identifier, e.g. 'main', 'branch_01'")
    from_beat_id: Optional[str] = Field(
        None,
        description="Beat ID where this branch conceptually diverges (None for main route)",
    )
    from_scene_id: Optional[str] = Field(
        None,
        description="Canonical scene ID where the player makes the branching choice",
    )
    kind: Literal["route", "ending"] = Field(
        "ending",
        description="Type: overarching route or purely ending variant",
    )
    title: str = Field(..., description="Short name of the branch (for UI)")
    description: str = Field(..., description="High-level description of this route/ending")
    ending_tone: Optional[str] = Field(
        None,
        description="Tone of the ending, e.g. 'good', 'bad', 'bittersweet'",
    )
    is_canonical: bool = Field(False, description="True only for main canonical route")


class BranchingInfo(BaseModel):
    max_branches: int = Field(1, description="Maximum number of branches requested")
    branches: List[BranchSpec] = Field(default_factory=list, description="Defined routes/endings")



class StoryState(BaseModel):

    world: Dict[str, Any] = Field(default_factory=dict, description="Global world-level facts")
    characters: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-character state: mood, location, relationship flags, etc.",
    )
    plot_threads: Dict[str, str] = Field(
        default_factory=dict,
        description="Story threads and their status: open/active/resolved/dropped",
    )



class PlotPreferences(BaseModel):
    hook: Optional[List[
        Literal[
            "случайная встреча", "переезд в новый город", "таинственное письмо",
            "необычное происшествие", "важное поручение", "начало путешествия",
            "потеря памяти", "пропажа человека", "неожиданное наследство",
            "семейный конфликт"
        ]
    ]] = Field(None, description="Предпочтительная завязка")

    main_branches: Optional[List[
        Literal[
            "романтическая линия", "детективная ветка", "мистическая линия",
            "политическая интрига", "линия дружбы", "тайна прошлого",
            "конфликт фракций", "путь героя"
        ]
    ]] = Field(None, description="Основные сюжетные ветки")

    key_player_choices: Optional[List[
        Literal[
            "выбор союзника/романтического интереса", "доверять или нет",
            "спасти или пожертвовать", "признаться в чувствах или скрыть",
            "выбрать сторону конфликта", "следовать правилам или нарушить",
            "милосердие или жесткость", "рискнуть или избегать опасности",
            "сказать правду или соврать", "применить способность или отказаться"
        ]
    ]] = Field(None, description="Ключевые выборы игрока")

    climax: Optional[List[
        Literal[
            "раскрытие главной тайны", "предательство", "столкновение с антагонистом",
            "шокирующее открытие о себе", "поворот сюжета",
            "выбор судьбы мира/группы", "ситуация на время",
            "конфликт союзников", "жертва героя или другого персонажа"
        ]
    ]] = Field(None, description="Варианты кульминации")

    resolution: Optional[List[
        Literal[
            "победа над антагонистом", "примирение сторон", "спасение/побег",
            "успешное выполнение миссии", "трагическая потеря", "разрыв отношений",
            "объединение группы", "возвращение к нормальной жизни",
            "начало нового пути", "открытый финал"
        ]
    ]] = Field(None, description="Варианты развязки")

    ending_types: Optional[List[
        Literal[
            "хэппи-энд", "плохая концовка", "нейтральная концовка",
            "персональные финалы для персонажей", "комическая концовка",
            "трагическая концовка", "романтические варианты финалов",
            "открытый финал"
        ]
    ]] = Field(None, description="Типы финалов")

    freeform: Optional[str] = Field(None, description="Произвольное текстовое описание желаемого сюжета")


class VNGenerationRequest(BaseModel):
    user_prompt: str = Field(..., description="User's story request")

    time_choice: Optional[Literal["древность", "средневековье", "современность"]] = Field(
        None, description="Выбор эпохи"
    )
    genre_choice: Optional[Literal["хоррор", "фентези", "фантастика", "повседневность", "романтика"]] = Field(
        None, description="Жанр"
    )
    tone_choice: Optional[Literal["веселый", "грустный"]] = Field(
        None, description="Тон истории (RU-домен)"
    )

    mc_name: Optional[str] = Field(None, description="Имя главного героя (обязательно)")
    mc_description: Optional[str] = Field(None, description="Описание ГГ (опционально)")

    extra_character_names: Optional[List[str]] = Field(
        None, description="Список имён второстепенных персонажей и/или антагониста"
    )

    plot_prefs: Optional[PlotPreferences] = Field(
        None, description="Структурированные сюжетные предпочтения"
    )
    plot_freeform: Optional[str] = Field(
        None, description="Свободное поле для описания сюжета"
    )

    graphic_style: Optional[Literal["аниме", "реализм", "рисованная графика"]] = Field(
        None, description="Графический стиль"
    )

    story_length: Optional[str] = Field("medium", description="short, medium, or long")
    max_branches: Optional[int] = Field(3, description="Maximum story branches (routes)")
    tone: Optional[str] = Field("balanced", description="Story tone (EN domain)")
    artstyle: Optional[str] = Field("anime", description="Art style (EN domain)")
    generate_images: Optional[bool] = Field(
        True,
        description="If false, skip all image generation (sprites/backgrounds).",
    )

    char_list: Optional[List[str]] = Field(None, description="Predefined character list (overrides user fields)")
    loc_list: Optional[List[str]] = Field(None, description="Predefined location list")
    setting_override: Optional[str] = Field(None, description="Override setting")


class VNGenerationResponse(BaseModel):
    status: str = Field(..., description="Generation status")
    message: str = Field(..., description="Status message")
    generation_id: str = Field(..., description="Unique generation ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Complete generation context")
    error: Optional[str] = Field(None, description="Error message if failed")