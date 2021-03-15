Column | Description | Dtype
--- | --- | ---
backers_count | Number of contributors | int
blurb | Description of the project | object
category | Category of the project | object: dictionary with category ID, name, slug, ...
converted_pledged_amount | Pledged amount the project got after the campaign finished | int
country | 2 Letter ISO Code | categorical
created_at | Time of creation of the project | datetime
creator | Information about the creator | object: dict with further info
currency | Abbreviation for currency | categorical: str
currency_symbol | Symbol of the currency | categorical: str
currency_trailing_code | Whether the currency symbol leading or trailing the value | categorical: bool
current_currency | All amounts are stated in current currency | categorical: str
deadline | Deadline for reaching the goal | datetime
disable_communication | Communication with the project creator is disabled for suspended projects| categorical: boolean
friends | ??? | NaN or [] 
fx_rate | Foreign exchange rate (USD to local currency) | float
goal | Target amount (in local currency) | 
id | Kickstarter unique ID | int
is_backing ||
is_starrable | A project can be starred while it is live | categorical: boolean
is_starred | Is the project starred | categorical: boolean
launched_at | When was the project launched | datetime
location | Location information on the project | object: dict with further info
name | Name of the project | object: str
permissions | ??? | NaN or []
photo | Link to the picture | object: dict with `key`, `full`, `med` ... as keys
pledged | Amount pledged (in local currency) | float
profile | Description of the profile used for the project (?) | object: dict with further details
slug | Slug-name-of-the-project | object: str
source_url | Url to the category of the project | object: str
spotlight | Allows creators to make a home for their project on kickstarter after they have been successfully funded | categorical: boolean
staff_pick | Whether or not the project was picked by the staff due to a really fun video, creative and well-priced rewards, a great story or an exciting idea | categorical: boolean
state | Status of the project: `canceled` - by the creator; `failed` - did not reach the goal before the deadline; `live` - open project; `successful` - successful project; `suspended` - A project may be suspended if Kickstarter's Trust & Safety team uncovers evidence that it is in violation of Kickstarter's rules | categorical: str
state_changed_at | Time of last state change | datetime
static_usd_rate | Rate to convert amounts to USD | float
urls | Collection of urls | object: dict with `web` and `rewards`
usd_pledged | Amount pledged (in USD) | float
usd_type | either `domestic`, `international`, or `NaN` | categorical: str
